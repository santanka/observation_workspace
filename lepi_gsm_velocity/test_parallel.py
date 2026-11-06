"""Partition isolation, resume equivalence and actual subprocess locking."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import convert
import parallel
import test_convert


class ParallelTests(unittest.TestCase):
    def test_partition_keeps_whole_days_and_limits_workers(self):
        start = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        items = [(start + dt.timedelta(days=i // 2, minutes=i % 2), Path(str(i))) for i in range(8)]
        parts = parallel.partition_days(items, 3)
        self.assertEqual(len(parts), 3)
        days = [day for part in parts for day, _ in part]
        self.assertEqual(len(days), len(set(days)))
        self.assertEqual(sum(len(paths) for part in parts for _, paths in part), 8)
        self.assertEqual(len(parallel.partition_days(items, 20)), 4)
        self.assertEqual(parallel.partition_days([], 4), [])

    def test_partition_worker_matches_serial_and_reuses_output(self):
        fixture = test_convert.Tests()
        fixture.setUp()
        try:
            fixture.config['earth_radius_km'] = 6371.2
            job = {'runtime': str(fixture.base / 'runtime'), 'worker': 0, 'cpu': None,
                   'config': fixture.config, 'moments_root': str(fixture.source), 'root': str(fixture.target),
                   'identity': 'gsm', 'min_free_memory_gib': 0., 'pause_seconds': 0.,
                   'days': [('1970-01-01T00:00:00+00:00', [str(fixture.path)])]}
            fixture.run_chunk()
            with np.load(fixture.output, allow_pickle=False) as saved:
                reference = {k: saved[k].copy() for k in saved.files}
            fixture.output.unlink()
            support = fixture.support
            support.ensure = lambda day: None
            def rotate(*args, **kwargs):
                import core
                return core.rotation(*args, **kwargs, transform=lambda **kw: kw['data_in'])
            with patch.object(convert, 'Support', return_value=support), patch.object(convert, 'rotation', side_effect=rotate), \
                 patch.object(parallel.os, 'nice'), patch.object(parallel.signal, 'signal'):
                self.assertEqual(parallel.process_partition(job, convert), 0)
                with np.load(fixture.output, allow_pickle=False) as saved:
                    for key, value in reference.items():
                        np.testing.assert_array_equal(saved[key], value)
                digest = convert.sha(fixture.output)
                self.assertEqual(parallel.process_partition(job, convert), 0)
                self.assertEqual(convert.sha(fixture.output), digest)
            status = json.loads((Path(job['runtime']) / 'status.json').read_text())
            self.assertEqual(status['counts'], {'skipped_verified': 1})
        finally:
            fixture.tearDown()

    def test_subprocesses_inherit_lock_and_run_concurrently(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            worker = base / 'worker.py'
            worker.write_text('''import fcntl, json, os, pathlib, sys, time
path=pathlib.Path(sys.argv[1])
os.fstat(int(sys.argv[2]))
while not (path.parent/'closed').exists(): time.sleep(.01)
with open(sys.argv[3],'a+') as check:
 try:
  fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  pass
 else:
  raise RuntimeError('Inherited flock was lost')
start=time.time()
time.sleep(.3)
path.write_text(json.dumps([start,time.time()]))
''')
            paths = [base / 'a.json', base / 'b.json']
            with (base / 'lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fd = lock.fileno()
                def close_parent_lock():
                    lock.close()
                    (base / 'closed').touch()
                codes = parallel.supervise(paths, fd, close_parent_lock,
                                           lambda p: [sys.executable, str(worker), str(p), str(fd), str(base / 'lock')])
                self.assertEqual(codes, [0, 0])
            with (base / 'lock').open('a+') as released:
                fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)
            spans = [json.loads(path.read_text()) for path in paths]
            self.assertLess(max(s[0] for s in spans), min(s[1] for s in spans))


if __name__ == '__main__':
    unittest.main()
