"""Offline parallel driver using the unchanged, profile-compatible GSM converter."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent


def process_partition(job, converter=None):
    # Set before importing convert/SpacePy; every worker has its own writable config.
    runtime = Path(job['runtime'])
    for name, folder in [('SPACEPY', 'spacepy'), ('MPLCONFIGDIR', 'matplotlib')]:
        directory = runtime / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    if converter is None:
        import convert as converter
        from attitude_support import Support
    else:
        Support = converter.Support
    c = converter
    os.nice(19)
    if hasattr(os, 'sched_setaffinity') and job.get('cpu') is not None:
        os.sched_setaffinity(0, {job['cpu']})
    support = Support(SimpleNamespace(offline=True, download_attempts=1, retry_delay=0.,
                                        earth_radius_km=job['config']['earth_radius_km']))
    counts = Counter()
    report = {'status': 'running', 'worker': job['worker'], 'counts': {}, 'processed_days': 0}
    status_path = runtime / 'status.json'
    def publish():
        report.update(counts=dict(counts), updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        c.atomic_json(status_path, report)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    publish()
    try:
        for day_text, items in job['days']:
            day = c.parse(day_text)
            failed_support = None
            for text in items:
                state_path = Path(text)
                relative = state_path.relative_to(job['moments_root'])
                try:
                    while c.available_gib() < job['min_free_memory_gib']:
                        time.sleep(10)
                    if failed_support is not None:
                        raise RuntimeError(failed_support)
                    try:
                        support.ensure(day)
                    except Exception as exc:
                        failed_support = str(exc)
                        raise
                    status = c.convert_chunk(state_path, Path(job['moments_root']), Path(job['root']),
                                             job['identity'], job['config'], support)
                    if status in ('partial', 'skipped_partial'):
                        state = json.loads((Path(job['root']) / relative).read_text())
                        if state.get('rotation_incomplete', True):
                            counts['rotation_incomplete'] += 1
                except Exception as exc:
                    status = 'failed'
                    source = json.loads(state_path.read_text())
                    c.atomic_json(Path(job['root']) / relative,
                                  {'status': status, 'profile_hash': job['identity'], 'reason': str(exc),
                                   'source_state': str(state_path), 'start_utc': source['start_utc'],
                                   'end_utc_exclusive': source['end_utc_exclusive'],
                                   'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
                    print(f"worker {job['worker']} {state_path.stem} failed: {exc}", flush=True)
                counts[status] += 1
                report['last_chunk'] = state_path.stem
                # Zero by default; no deliberate pauses on verified skips.
                if job['pause_seconds'] and status not in ('skipped_verified', 'skipped_partial'):
                    time.sleep(job['pause_seconds'])
                if sum(counts.get(k, 0) for k in ('complete', 'partial', 'failed', 'skipped_verified', 'skipped_partial')) % 20 == 0:
                    publish()
            report['processed_days'] += 1
            publish()
            print(f"worker {job['worker']} {day_text[:10]}: {dict(counts)}", flush=True)
        c.check_code(job['config'])
        report['status'] = 'finished'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        publish()
        return 130
    except BaseException:
        report['status'] = 'crashed'
        publish()
        raise
    publish()
    return 2 if counts['failed'] or counts['rotation_incomplete'] else 0


def partition_days(items, workers):
    grouped = {}
    for lo, path in items:
        day = lo.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        grouped.setdefault(day, []).append(str(path))
    parts = [[] for _ in range(min(workers, len(grouped)))]
    for i, item in enumerate(sorted(grouped.items())):
        parts[i % len(parts)].append(item)
    return parts


def supervise(job_paths, lock_fd, update, command=None):
    """Children inherit the parent's flock, retaining it even if the parent dies."""
    children = []
    try:
        for path in job_paths:
            argv = command(path) if command else [sys.executable, str(BASE / 'parallel.py'), '--worker-job', str(path)]
            children.append(subprocess.Popen(argv, pass_fds=(lock_fd,), start_new_session=True))
        while any(child.poll() is None for child in children):
            update()
            time.sleep(1)
        update()
        return [child.returncode for child in children]
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--moments-root', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--month'); group.add_argument('--year', type=int); group.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--output-root', type=Path, default=Path('/mnt/j/observation_data/Arase_LEPi_GSM_velocity'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--offline', action='store_true', default=True, help='All workers are always offline')
    parser.add_argument('--pause-seconds', type=float, default=0.)
    parser.add_argument('--min-free-memory-gib', type=float, default=8.)
    parser.add_argument('--max-attitude-gap', type=float, default=16.)
    parser.add_argument('--max-orbit-gap', type=float, default=120.)
    parser.add_argument('--earth-radius-km', type=float, default=6371.2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.worker_job:
        return process_partition(json.loads(args.worker_job.read_text()))
    if not args.moments_root or not any((args.month, args.year, args.start)):
        parser.error('--moments-root and --month/--year/--start are required')
    import math
    if args.workers < 1 or any(not math.isfinite(v) or v < 0 for v in
                             (args.pause_seconds, args.min_free_memory_gib)) or any(
                             not math.isfinite(v) or v <= 0 for v in
                             (args.max_attitude_gap, args.max_orbit_gap, args.earth_radius_km)):
        parser.error('Invalid worker, resource or interpolation settings')
    os.environ['SPACEPY'] = f'/tmp/lepi_gsm_parallel_parent_{os.getpid()}'
    import convert as c
    args.moments_root = args.moments_root.resolve()
    args.output_root = args.output_root.resolve()
    start, end = c.selected_range(args)
    source = json.loads((args.moments_root / 'profile.json').read_text())
    identity, config = c.profile(args, source)
    root = args.output_root / ('gsm_observed_' + identity[:12])
    items = c.inventory(args.moments_root, start, end)
    parts = partition_days(items, args.workers)
    report = {'profile_hash': identity, 'source_start_utc': start.isoformat(),
              'source_end_utc_exclusive': end.isoformat(), 'source_usable_chunks': len(items),
              'workers': len(parts), 'requested_workers': args.workers, 'download_workers': 0,
              'offline': True, 'pause_seconds': args.pause_seconds,
              'execution_sources': {str(BASE / name): c.sha(BASE / name)
                                    for name in ('parallel.py', 'attitude_support.py')},
              'inventory_scope': 'Verified complete/partial source chunks present at startup'}
    if args.dry_run:
        print(json.dumps({**report, 'output_root': str(root)}, indent=2)); return 0
    if not parts:
        print('No source chunks selected'); return 2
    root.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    with (root / '.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A converter holds this profile lock. Stop the existing run with Ctrl+C before restarting.', file=sys.stderr)
            return 3
        if (root / 'profile.json').exists() and json.loads((root / 'profile.json').read_text()) != {'profile_hash': identity, 'config': config}:
            raise ValueError('Existing profile differs')
        c.atomic_json(root / 'profile.json', {'profile_hash': identity, 'config': config})
        runtime = Path(tempfile.mkdtemp(prefix='lepi_gsm_parallel_', dir='/tmp'))
        report.update(status='running', runtime=str(runtime), output_root=str(root))
        jobs = []
        cpus = sorted(os.sched_getaffinity(0))
        for i, days in enumerate(parts):
            directory = runtime / str(i)
            directory.mkdir()
            job = {**vars(args), 'moments_root': str(args.moments_root.resolve()), 'root': str(root.resolve()),
                   'config': config, 'identity': identity, 'worker': i, 'cpu': cpus[i % len(cpus)],
                   'runtime': str(directory), 'days': days}
            # Only the worker settings used above are serialized; no pathlib objects.
            for key in ('output_root', 'worker_job'):
                job.pop(key, None)
            path = directory / 'job.json'
            c.atomic_json(path, job); jobs.append(path)
        def publish():
            counts = Counter(); statuses = []
            for job_path in jobs:
                path = job_path.parent / 'status.json'
                if path.exists():
                    status = json.loads(path.read_text())
                    counts.update(status['counts']); statuses.append(status)
            report.update(counts=dict(counts), worker_statuses=statuses,
                          updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            c.atomic_json(root / 'parallel_last_run.json', report)
            c.atomic_json(root / 'last_run.json', report)
        try:
            publish()
            codes = supervise(jobs, lock.fileno(), publish)
            report.update(status='finished' if all(code in (0, 2) for code in codes) else 'failed',
                          worker_exit_codes=codes)
            c.check_code(config)
        except KeyboardInterrupt:
            report['status'] = 'interrupted'
            publish()
            return 130
        except BaseException:
            report['status'] = 'failed'
            publish()
            raise
        publish()
        print(json.dumps({'output_root': str(root), 'counts': report['counts'], 'worker_exit_codes': codes}))
        return 0 if all(code == 0 for code in codes) else 2


if __name__ == '__main__':
    raise SystemExit(main())
