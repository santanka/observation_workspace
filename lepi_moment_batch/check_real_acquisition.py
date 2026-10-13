"""Explicit small smoke test: one LEP-i day (reuse raw if verified), one fresh MGF hour, 5 min."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from acquisition import Acquisition, REMOTE
import batch


def main():
    started = time.monotonic()
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    args = argparse.Namespace(output_root=batch.DEFAULT_OUTPUT, input_root=batch.DEFAULT_INPUT,
                remote_root=REMOTE, offline=False, refresh_downloads=False,
                download_attempts=2, retry_delay=1., http_timeout=15., download_timeout=120.,
                validation_timeout=180., min_free_memory_gib=8., scratch_reserve_gib=4.,
                mgf_cadence='64hz', max_mgf_gap_seconds=1.)
    identity, config = batch.profile(args)
    scratch = Path('/tmp/lepi_moments_work')/identity[:12]
    scratch.mkdir(parents=True, exist_ok=True)
    date = batch.parse_utc('2022-09-29')
    a = Acquisition(args, scratch)
    inputs = [a.ensure('lepi', date)]
    # Fresh actual GET only for the small (~5 MiB) magnetic granule.
    args.refresh_downloads = True
    inputs.append(a.ensure('64hz', date.replace(hour=5)))
    record = {'profile_hash': identity, 'input_statuses': [x['status'] for x in inputs],
              'input_origins': [x.get('origin') for x in inputs],
              'input_sha256': [x.get('sha256') for x in inputs],
              'scope': 'One day LEP-i validation, one hour MGF transfer, five-minute moments only'}
    destination = args.output_root/'_validation/acquisition_verification.json'
    if any(x['status'] != 'verified' for x in inputs):
        batch.atomic_json(destination, record)
        raise RuntimeError('Actual acquisition did not verify both required granules')
    batch.check_sources(config)
    command = [sys.executable, str(batch.BASE/'batch.py'), '--start', '2022-09-29T05:05:00',
               '--end', '2022-09-29T05:10:00', '--offline', '--pause-seconds', '0']
    reports = []
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, timeout=240)
        print(result.stdout, end='')
        if result.returncode:
            raise RuntimeError(result.stderr + result.stdout)
        reports.append(json.loads(result.stdout.splitlines()[-1]))
    assert reports[-1]['counts'] == {'skipped_verified': 1}, reports
    record['runs'] = reports
    import numpy as np
    root = Path(reports[0]['output_root'])
    output = root/'2022/09/data/20220929/20220929_050500_20220929_051000.npz'
    reference = args.output_root/'_validation/pilot_300s.npz'
    if reference.exists():
        with np.load(reference, allow_pickle=False) as old, np.load(output, allow_pickle=False) as new:
            keys = set(old.files) - {'manifest_json'}
            for key in keys:
                np.testing.assert_array_equal(old[key], new[key], err_msg=key)
            record['reference_arrays_exactly_equal'] = len(keys)
    record['total_seconds'] = time.monotonic() - started
    batch.atomic_json(destination, record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
