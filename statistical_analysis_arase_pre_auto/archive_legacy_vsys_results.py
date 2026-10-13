"""Archive the spacecraft-subtracted velocity analysis (dry run by default).

Move each dataset under auto/run_state on the same filesystem and retain compatibility
symlinks so paths in archived markers/CSV files remain valid.
Raw observations, wavelet caches, and range manifests are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation')
OLD_VERSION = 'lepe_ge67eV_v1'
NEW_VERSION = 'lepe_ge67eV_vion_v2'
ARCHIVE = ROOT / 'legacy' / 'lepe_ge67eV_v1_vsys_subtracted_20261009'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    processes = subprocess.check_output(['ps', '-eo', 'comm=,args='], text=True)
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].startswith('python'):
            if ('batch_arase.py' in fields[1]
                    or 'statistical_analysis_arase_pre_auto.batch_arase' in fields[1]
                    or 'statistical_analysis_arase_pre_auto/plot_' in fields[1]):
                raise RuntimeError(f'Active analysis/plot process: {line.strip()}')
    version_root = ROOT / OLD_VERSION
    plan = []
    # Windows can hold the auto/run_state parent directories open even after
    # WSL batch processes exit. Dataset renames preserve their parent paths.
    for name in ('auto', 'run_state'):
        parent = version_root / name
        for source in sorted(parent.iterdir()):
            target = ARCHIVE / name / source.name
            if source.is_symlink():
                if source.resolve() != target or not target.is_dir():
                    raise RuntimeError(f'Unexpected symlink: {source}')
                state = 'already_archived'
            else:
                if not source.is_dir() or target.exists():
                    raise RuntimeError(f'Unexpected source/destination: {source} -> {target}')
                state = 'pending'
            plan.append({'source': str(source), 'archive': str(target), 'state': state})
    datasets = sorted(p.name for p in (version_root / 'run_state').iterdir() if p.is_dir())
    migration = {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'old_analysis_version': OLD_VERSION,
        'new_analysis_version': NEW_VERSION,
        'old_system_velocity_policy': 'ion_moment_minus_spacecraft_velocity',
        'new_system_velocity_policy': 'observed_partial_ion_moment_no_spacecraft_subtraction',
        'paths': plan,
        'datasets': datasets,
        'raw_observations_moved': False,
        'reanalysis_started': False,
    }
    print(json.dumps(migration, indent=2))
    if not args.apply:
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    failures = []
    for entry in plan:
        if entry['state'] == 'already_archived':
            continue
        source, target = Path(entry['source']), Path(entry['archive'])
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(source, target)
        except PermissionError:
            entry['state'] = 'rename_blocked'
            failures.append(str(source))
            print(f'LOCKED: {source}', flush=True)
            continue
        try:
            source.symlink_to(target, target_is_directory=True)
        except Exception:
            os.rename(target, source)
            raise
        entry['state'] = 'archived'
        print(f'Archived {source} -> {target}; compatibility symlink installed')
    migration['migration_complete'] = not failures
    migration['pending_paths'] = failures
    (ARCHIVE / 'migration.json').write_text(json.dumps(migration, indent=2) + '\n')
    if failures:
        raise RuntimeError('Directories could not be renamed: ' + ', '.join(failures))



if __name__ == '__main__':
    main()
