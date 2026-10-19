"""Fast failure paths, timestamp-cache invalidation and audited result reuse."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import batch


class SpeedupTests(unittest.TestCase):
    def test_lightweight_worker_and_cache(self):
        # Separate process: importing pySPEDAS elsewhere must not hide eager imports.
        script = '''
import json, sys, tempfile
from pathlib import Path
import worker
from test_acquisition import fixture
from batch import fingerprint
np = worker.np
assert 'pyspedas' not in sys.modules and 'pilot' not in sys.modules
with tempfile.TemporaryDirectory() as directory:
    p = Path(directory)
    raw = p/'lepi.cdf'
    fixture(raw, product='lepi')
    source = {'expanded_path': str(raw), 'source': fingerprint(raw), 'time_cache_path': str(p/'epochs.npz')}
    lo = np.datetime64('2022-09-29T05:10:00', 'ns')
    req = {'input_verification': 'verified', 'config': {}, 'lepi': [source], 'mgf': [],
           'start_utc': '2022-09-29T05:10:00+00:00', 'end_utc_exclusive': '2022-09-29T05:15:00+00:00'}
    assert worker.execute(req)['status'] == 'excluded'
    assert Path(source['time_cache_path']).exists()
    class NoRead:
        def varget(self, name):
            raise AssertionError('Epoch CDF read repeated')
    cached = worker.times(NoRead(), 'Epoch', source)
    actual = worker.times(worker.CDF(raw), 'Epoch')
    for a, b in zip(cached, actual): np.testing.assert_array_equal(a, b)
    fixture(raw, product='lepi', seconds=(0, 8, 16))
    with __import__('unittest.mock', fromlist=['patch']).patch.object(worker, 'fingerprint', return_value={'changed': True}):
        try: worker.times(NoRead(), 'Epoch', source)
        except AssertionError: pass
        else: raise AssertionError('Changed source reused stale timestamps')
    source['source'] = fingerprint(raw)
    req.update(start_utc='2022-09-29T05:05:00+00:00', end_utc_exclusive='2022-09-29T05:10:00+00:00',
               config={'mgf_cadence': '64hz', 'max_mgf_gap_seconds': 1.})
    try: worker.execute(req)
    except worker.MissingMagneticData: pass
    else: raise AssertionError('Missing field accepted')
    assert 'pyspedas' not in sys.modules and 'pilot' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', script], cwd=batch.BASE, check=True, timeout=30)

    def test_compatible_profile_and_reuse(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            old_config = {'conditions': 'unchanged', 'sources': dict(batch.REUSABLE_RUNNER_SOURCES)}
            old_id = hashlib.sha256(json.dumps(old_config, sort_keys=True).encode()).hexdigest()
            old_root = base / ('native_dsi_' + old_id[:12])
            batch.atomic_json(old_root/'profile.json', {'profile_hash': old_id, 'config': old_config})
            config = {**old_config, 'sources': {p: 'new reviewed implementation' for p in old_config['sources']}}
            previous = batch.compatible_reuse_root(old_root, config)
            with self.assertRaises(ValueError):
                batch.compatible_reuse_root(old_root, {**config, 'conditions': 'different'})
            root = base/'new'
            state, output = root/'state/day/chunk.json', root/'data/day/chunk.npz'
            old_state, old_output = old_root/state.relative_to(root), old_root/output.relative_to(root)
            old_output.parent.mkdir(parents=True)
            values = np.array([1., float('nan'), 3.])
            np.savez_compressed(old_output, data=values, epoch_tt2000=np.arange(3,dtype=np.int64),
                                manifest_json=json.dumps({'profile_hash': old_id, 'config': old_config}))
            proof = [{'unit': 'lepi_day', 'status': 'verified', 'sha256': 'raw bytes',
                      'fingerprint': {'path': 'raw'}, 'validation_identity': 'old'}]
            batch.atomic_json(old_state, {'profile_hash': old_id, 'status': 'complete',
                              'input_signature': 'old signature', 'input_verification': 'verified',
                              'acquisition_evidence': proof, 'output_sha256': batch.sha_file(old_output)})
            before = batch.sha_file(old_output)
            common = {'input_verification': 'verified', 'acquisition_evidence': [{**proof[0], 'validation_identity': 'new'}]}
            bad = {**common, 'acquisition_evidence': [{**proof[0], 'sha256': 'changed raw bytes'}]}
            self.assertFalse(batch.reuse_result(previous, root, state, output, 'new', 'signature', bad, config))
            self.assertTrue(batch.reuse_result(previous, root, state, output, 'new', 'signature', common, config))
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['data'], values)
                manifest = json.loads(str(saved['manifest_json']))
                self.assertEqual(manifest['profile_hash'], 'new')
                self.assertEqual(manifest['config'], config)
                np.testing.assert_array_equal(saved['moment_valid'], [True,True,True])
                np.testing.assert_array_equal(saved['moment_invalid_reason'], [0,0,0])
                self.assertEqual(manifest['reused_from']['output_sha256'], before)
            self.assertEqual(batch.sha_file(old_output), before)
            self.assertTrue(batch.resume_valid(state, output, 'new', 'signature'))
            self.assertFalse(batch.reuse_result(previous, root, state, output, 'new', 'signature', common, config))
            output.unlink()
            prior = json.loads(old_state.read_text())
            batch.atomic_json(old_state, {**prior, 'status': 'missing_mgf'})
            self.assertFalse(batch.reuse_result(previous, root, state, output, 'new', 'signature', common, config))
            batch.atomic_json(old_state, {**prior, 'status': 'no_data'})
            old_output.unlink()
            self.assertFalse(batch.reuse_result(previous, root, state, output, 'new', 'signature', common, config))


if __name__ == '__main__':
    unittest.main()
