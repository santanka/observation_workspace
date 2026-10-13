"""Sequential real-data pilot checks. No download and no KAW state changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'


def run(label, start, seconds, standard=False):
    output = RESULTS / (label + '.npz')
    cmd = ['ionice', '-c', '3', sys.executable, str(BASE / 'pilot.py'),
           '--start', start, '--seconds', str(seconds), '--output', str(output)]
    if standard:
        cmd.append('--standard-check')
    with (RESULTS / (label + '.log')).open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    metrics = json.loads(output.with_suffix('.metrics.json').read_text())
    print(json.dumps({k: metrics[k] for k in ['output', 'records_per_species', 'total_seconds',
                                             'max_rss_mib', 'file_bytes']}), flush=True)
    return output, metrics


def main():
    RESULTS.mkdir(exist_ok=True)
    cases = [('direct_60s', '2022-09-29T05:05:00', 60),
             ('direct_300s', '2022-09-29T05:05:00', 300),
             ('direct_900s', '2022-09-29T05:05:00', 900),
             ('chunk2_300s', '2022-09-29T05:10:00', 300),
             ('chunk3_300s', '2022-09-29T05:15:00', 300)]
    results = [run(*case) for case in cases]
    report = {'cases': [m for _, m in results], 'checks': {}}
    # Independent whole-variable standard-importer match was checked in this run.
    with np.load(RESULTS / 'standard_60s.npz', allow_pickle=False) as std, \
            np.load(results[0][0], allow_pickle=False) as direct:
        for key in std.files:
            if key != 'manifest_json':
                np.testing.assert_array_equal(std[key], direct[key], err_msg=key)
        report['checks']['standard_vs_direct_all_arrays'] = 'exact equality including NaN positions'
    with np.load(results[2][0], allow_pickle=False) as whole:
        chunks = [np.load(results[i][0], allow_pickle=False) for i in [1, 3, 4]]
        try:
            joined = np.concatenate([c['epoch_tt2000'] for c in chunks])
            np.testing.assert_array_equal(joined, whole['epoch_tt2000'])
            assert len(np.unique(joined)) == len(joined)
            tested = []
            for key in whole.files:
                if '__' in key and key.split('__')[-1] not in ['energy_kev_per_q', 'energy_selected']:
                    np.testing.assert_array_equal(np.concatenate([c[key] for c in chunks]),
                                                  whole[key], err_msg=key)
                    tested.append(key)
            report['checks']['three_chunks_vs_900s'] = {
                'records_per_species': len(joined), 'array_series_compared': len(tested),
                'result': 'exact equality; no duplicated or omitted boundary records'}
        finally:
            for c in chunks:
                c.close()
    # Same command must skip an existing file without rewriting it.
    output = results[1][0]
    before = output.stat().st_mtime_ns
    cmd = ['ionice', '-c', '3', sys.executable, str(BASE / 'pilot.py'), '--seconds', '300',
           '--output', str(output)]
    skip = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=180)
    assert json.loads(skip.stdout)['status'] == 'skipped_verified'
    assert output.stat().st_mtime_ns == before
    report['checks']['resume'] = 'configuration and array checksums verified; output not rewritten'
    # Changed configuration must reject an existing artifact, preserving its mtime.
    bad = subprocess.run(cmd[:-2] + ['--start', '2022-09-29T05:06:00', '--output', str(output)],
                         capture_output=True, text=True, timeout=180)
    assert bad.returncode != 0 and 'different configuration' in bad.stderr
    assert output.stat().st_mtime_ns == before
    report['checks']['incompatible_resume'] = 'rejected; existing artifact unchanged'
    (RESULTS / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['checks'], indent=2), flush=True)


if __name__ == '__main__':
    main()
