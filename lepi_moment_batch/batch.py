"""Verified-input monthly/yearly LEP-i moment runner; one bounded worker at a time."""
import argparse
from contextlib import contextmanager
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zlib

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
DEFAULT_INPUT = Path(os.environ.get('SPEDAS_DATA_DIR', '/mnt/j/observation_data')) / 'ergsc/satellite/erg'
DEFAULT_OUTPUT = Path('/mnt/j/observation_data/Arase_LEPi_moments')
UTC = dt.timezone.utc
CHUNK_SECONDS = 300


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def sha_file(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            sha.update(block)
    return sha.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    st = path.stat()
    return {'path': str(path), 'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def latest(directory, pattern):
    paths = list(directory.glob(pattern)) if directory.exists() else []
    def version(p):
        match = re.search(r'_v(\d+)[_.](\d+)\.cdf$', p.name)
        return tuple(map(int, match.groups())) if match else (-1, -1)
    return max(paths, key=version) if paths else None


def source_day(root, day):
    return latest(root / 'lepi/l2/3dflux' / day.strftime('%Y/%m'),
                  'erg_lepi_l2_3dflux_' + day.strftime('%Y%m%d') + '_v*.cdf')


def particle_sources(root, day, start, end):
    candidates = [day]
    if start == day:
        candidates.insert(0, day - dt.timedelta(days=1))
    if end == day + dt.timedelta(days=1):
        candidates.append(day + dt.timedelta(days=1))
    return [p for d in candidates if (p := source_day(root, d)) is not None]


def magnetic_sources(root, start, end, cadence):
    # Neighbouring files can supply a bracket at a UTC hour/day boundary.
    found = []
    if cadence == '64hz':
        cursor = (start - dt.timedelta(seconds=16)).replace(minute=0, second=0, microsecond=0)
        stop = end + dt.timedelta(seconds=16)
        step = dt.timedelta(hours=1)
        prefix = 'erg_mgf_l2_64hz_dsi_'
        fmt = '%Y%m%d%H'
    else:
        cursor = (start - dt.timedelta(seconds=32)).replace(hour=0, minute=0, second=0, microsecond=0)
        stop = end + dt.timedelta(seconds=32)
        step = dt.timedelta(days=1)
        prefix = 'erg_mgf_l2_8sec_'
        fmt = '%Y%m%d'
    while cursor <= stop:
        p = latest(root / 'mgf/l2' / cadence / cursor.strftime('%Y/%m'),
                   prefix + cursor.strftime(fmt) + '_v*.cdf')
        if p and p not in found:
            found.append(p)
        cursor += step
    return found


def parse_utc(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def requested_range(args):
    if args.month:
        if not re.fullmatch(r'\d{4}-\d{2}', args.month):
            raise ValueError('--month must be YYYY-MM')
        start = parse_utc(args.month + '-01')
        end = (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    elif args.year is not None:
        if not 1 <= args.year <= 9998:
            raise ValueError('--year must be 1–9998')
        start = dt.datetime(args.year, 1, 1, tzinfo=UTC)
        end = dt.datetime(args.year + 1, 1, 1, tzinfo=UTC)
    else:
        if not args.end:
            raise ValueError('--start requires --end')
        start, end = parse_utc(args.start), parse_utc(args.end)
    if args.end and not args.start:
        raise ValueError('--end is only valid with --start')
    if end <= start:
        raise ValueError('end must be after start')
    # Stable five-minute filenames; partial requests cannot overlap full-grid chunks.
    for point in [start, end]:
        if point.second or point.microsecond or point.minute % 5:
            raise ValueError('Range endpoints must align with the UTC five-minute grid')
    return start, end


def days(start, end):
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        next_day = cursor + dt.timedelta(days=1)
        yield cursor, max(start, cursor), min(end, next_day)
        cursor = next_day


def chunks(start, end):
    cursor = start
    while cursor < end:
        next_time = min(cursor + dt.timedelta(seconds=CHUNK_SECONDS), end)
        yield cursor, next_time
        cursor = next_time


def profile(args):
    package_versions = {k: importlib.metadata.version(k)
                        for k in ['numpy', 'scipy', 'xarray', 'pyspedas', 'cdflib']}
    package = Path(importlib.metadata.distribution('pyspedas').locate_file('pyspedas'))
    leaves = ['erg_lep_part_products', 'erg_lepi_get_dist', 'erg_pgs_clean_data',
              'erg_pgs_limit_range', 'erg_convert_flux_units', 'get_lepi_flux_angle_in_sga',
              'erg_pgs_moments_tplot']
    paths = [BASE / 'batch.py', BASE / 'worker.py', BASE / 'acquisition.py',
             BASE / 'validate_input.py', WORKSPACE / 'lepi_moment_pilot/pilot.py']
    paths += [package / ('projects/erg/satellite/erg/particle/' + n + '.py') for n in leaves]
    paths += [package / ('particles/moments/' + n + '.py')
              for n in ['spd_pgs_moments', 'moments_3d', 'moments_3d_omega_weights']]
    paths += [package / 'analysis/tinterpol.py', package / 'utilities/tcopy.py',
              package / 'tplot_tools/store_data.py']
    paths += [Path(importlib.metadata.distribution('cdflib').locate_file('cdflib')) / n
              for n in ['cdfread.py', 'epochs.py']]
    config = {'schema': 'lepi_monthly_moments_v4_shared_cache_prefetch', 'chunk_seconds': CHUNK_SECONDS,
              'energy_range_ev_per_q': [30., 30000.],
              'species': {'FPDU': 'proton', 'FHEDU': 'heplus', 'FODU': 'oplus'},
              'frame': 'native DSI; spacecraft-relative; no spacecraft-velocity correction',
              'spacecraft_potential_volts_used': 0., 'quality_mask_applied': False,
              'lepi_boundary_tolerance_seconds': getattr(args, 'lepi_boundary_tolerance_seconds', 64.),
              'duplicate_record_policy': 'within granule: identical payload only, retain last; across granules: last dated source',
              'sub_anodes_used': False, 'mgf_cadence': args.mgf_cadence,
              'max_mgf_gap_seconds': args.max_mgf_gap_seconds,
              'packages': package_versions,
              'sources': {str(p): sha_file(p) for p in paths}}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return identity, config


def memory_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('MemAvailable is unavailable')


def check_sources(config):
    for path, expected in config['sources'].items():
        if sha_file(path) != expected:
            raise RuntimeError(f'Calculation source changed during run: {path}')


def wait_resources(args, scratch, expanded_size=0):
    announced = False
    while memory_available_gib() < args.min_free_memory_gib:
        if not announced:
            print(f'waiting: available memory below {args.min_free_memory_gib:g} GiB', flush=True)
            announced = True
        time.sleep(10)
    if shutil.disk_usage(scratch).free < expanded_size + args.scratch_reserve_gib * 1024**3:
        raise RuntimeError('Insufficient scratch disk space after expansion reserve')


def stage_cdf(source, cache, args):
    """Stream CDF3 gzip once per source per day, without importing cdflib."""
    source = Path(source).resolve()
    source_fp = fingerprint(source)
    with source.open('rb') as f:
        head = f.read(40)
        if head[4:8] == bytes.fromhex('0000ffff'):
            return {'source': source_fp, 'expanded_path': str(source), 'compressed': False}
        if head[:4] != bytes.fromhex('cdf30001'):
            raise ValueError('Only CDF3 file-level gzip is supported')
        block_size = int.from_bytes(head[8:16], 'big')
        cpr = int.from_bytes(head[20:28], 'big')
        expected = int.from_bytes(head[28:36], 'big')
        f.seek(cpr + 12)
        if int.from_bytes(f.read(4), 'big') != 5:
            raise ValueError('Unsupported CDF compression type')
    key = hashlib.sha256(json.dumps(source_fp, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target, marker = cache / (key + '.cdf'), cache / (key + '.json')
    if target.exists() and marker.exists():
        try:
            meta = json.loads(marker.read_text())
            if (meta['source'] == source_fp and target.stat().st_size == expected + 8
                    and sha_file(target) == meta['expanded_sha256']):
                return meta
        except (OSError, ValueError, KeyError):
            pass
        target.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
    wait_resources(args, cache, expected + 8)
    fd, temp = tempfile.mkstemp(prefix=key + '.', suffix='.tmp', dir=cache)
    sha = hashlib.sha256()
    count = 0
    try:
        with source.open('rb') as inp, os.fdopen(fd, 'wb') as out:
            header = bytes.fromhex('cdf300010000ffff')
            out.write(header)
            sha.update(header)
            inp.seek(40)
            remaining = block_size - 32
            dec = zlib.decompressobj(31)
            while remaining:
                block = inp.read(min(1024**2, remaining))
                if not block:
                    raise EOFError('Truncated compressed CDF')
                remaining -= len(block)
                while block:
                    expanded = dec.decompress(block, 1024**2)
                    count += len(expanded)
                    out.write(expanded)
                    sha.update(expanded)
                    block = dec.unconsumed_tail
                if dec.unused_data or count > expected:
                    raise ValueError('Invalid gzip payload or expanded length')
            if not dec.eof or count != expected or fingerprint(source) != source_fp:
                raise ValueError('Gzip validation failed or source changed during staging')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, target)
        meta = {'source': source_fp, 'expanded_path': str(target), 'compressed': True,
                'expanded_bytes': count + 8, 'expanded_sha256': sha.hexdigest(),
                'gzip_crc_and_length_verified': True}
        atomic_json(marker, meta)
        return meta
    finally:
        Path(temp).unlink(missing_ok=True)


def resume_valid(state_path, output, identity, signature):
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text())
        if state.get('profile_hash') != identity or state.get('input_signature') != signature:
            return False
        if state['status'] == 'no_data':
            return state.get('input_verification') == 'verified' and not output.exists()
        return (state['status'] == 'complete' and state.get('input_verification') == 'verified' and output.exists()
                and state['output_sha256'] == sha_file(output))
    except (OSError, ValueError, KeyError):
        return False


def monthly_summary(month_dir):
    statuses = Counter()
    for path in (month_dir / 'state').glob('*/*.json'):
        try:
            statuses[json.loads(path.read_text())['status']] += 1
        except (ValueError, KeyError):
            statuses['invalid_state'] += 1
    # This is an inventory, not a claim of full-month observation coverage.
    atomic_json(month_dir / 'summary.json', {'updated_utc': dt.datetime.now(UTC).isoformat(),
                'chunk_state_counts': dict(statuses),
                'scope': 'Inventory of attempted chunks; does not imply a fully processed month'})


@contextmanager
def track_run(root, start, end, identity, counts):
    base = {'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
            'profile_hash': identity, 'output_root': str(root)}
    atomic_json(root / 'last_run.json', {**base, 'status': 'running', 'counts': dict(counts)})
    try:
        yield
    except BaseException as exc:
        atomic_json(root / 'last_run.json', {**base, 'status': 'interrupted' if isinstance(exc, KeyboardInterrupt)
                                           else 'stopped', 'reason': str(exc), 'exception': type(exc).__name__,
                                           'counts': dict(counts),
                                           'updated_utc': dt.datetime.now(UTC).isoformat()})
        raise
    else:
        atomic_json(root / 'last_run.json', {**base, 'status': 'finished', 'counts': dict(counts),
                                           'updated_utc': dt.datetime.now(UTC).isoformat()})


def run(args, start, end, identity, config):
    root = args.output_root / ('native_dsi_' + identity[:12])
    root.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch_root / identity[:12]
    scratch.mkdir(parents=True, exist_ok=True)
    from acquisition import Acquisition
    acquisition = Acquisition(args, scratch)
    from progress import Progress
    total = int((end - start).total_seconds()) // CHUNK_SECONDS
    counts = Counter()
    chunk_statuses = {}
    def record(label, status):
        previous = chunk_statuses.get(label)
        if previous is not None:
            counts[previous] -= 1
            if not counts[previous]:
                del counts[previous]
        chunk_statuses[label] = status
        counts[status] += 1
    with Progress(total, getattr(args, 'progress', 'auto')) as progress, (root / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A separate lock protects scratch even if an alternate output root is used.
        with (scratch / '.scratch.lock').open('a+') as scratch_lock:
            fcntl.flock(scratch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with track_run(root, start, end, identity, counts):
                atomic_json(root / 'profile.json', {'profile_hash': identity, 'config': config})
                selected = None
                retry_passes = 0 if args.offline else getattr(args, 'retry_passes', 1)
                for pass_number in range(retry_passes + 1):
                    if pass_number:
                        if not pending:
                            break
                        selected = pending
                        print(f'Retrying acquisition for {len(selected)} failed chunks (pass {pass_number})', flush=True)
                        time.sleep(getattr(args, 'retry_pass_delay', 30.))
                        acquisition.retry_failed()
                        progress.restart(len(selected))
                    pending = set()
                    prefetched = set()
                    for day, first, stop in days(start, end):
                        for package, expected in config['packages'].items():
                            if importlib.metadata.version(package) != expected:
                                raise RuntimeError(f'Package version changed during run: {package}')
                        if selected is not None and not any(first <= lo < stop for lo, _ in selected):
                            continue
                        month_key = day.strftime('%Y/%m')
                        if month_key not in prefetched:
                            if selected is None:
                                month_start = max(start, day.replace(day=1))
                                month_end = min(end, (day.replace(day=28) + dt.timedelta(days=4)).replace(day=1))
                                intervals = list(days(month_start, month_end))
                            else:
                                intervals = [(lo.replace(hour=0, minute=0, second=0, microsecond=0), lo, hi)
                                             for lo, hi in selected if lo.strftime('%Y/%m') == month_key]
                            progress.current(day, 'prefetch')
                            inventory = acquisition.prefetch(intervals, args.mgf_cadence, getattr(args, 'progress', 'auto'))
                            atomic_json(root / 'acquisition_passes' / f'pass_{pass_number}_{month_key.replace("/", "_")}.json',
                                        {'pass': pass_number, 'units': inventory,
                                         'updated_utc': dt.datetime.now(UTC).isoformat()})
                            prefetched.add(month_key)
                        month_dir = root / day.strftime('%Y/%m')
                        day_label = day.strftime('%Y%m%d')
                        cache = scratch / day_label
                        staged = {}
                        try:
                            for lo, hi in chunks(first, stop):
                                if selected is not None and (lo, hi) not in selected:
                                    continue
                                label = lo.strftime('%Y%m%d_%H%M%S') + '_' + hi.strftime('%Y%m%d_%H%M%S')
                                state_path = month_dir / 'state' / day_label / (label + '.json')
                                output = month_dir / 'data' / day_label / (label + '.npz')
                                check_sources(config)
                                progress.current(lo, 'acquire/verify')
                                particles, mag, proof, failures, primary = acquisition.interval(
                                    day, lo, hi, args.mgf_cadence)
                                inputs = [fingerprint(p) for p in particles + mag]
                                signature = hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()
                                common = {'start_utc': lo.isoformat(), 'end_utc_exclusive': hi.isoformat(),
                                          'profile_hash': identity, 'input_signature': signature, 'inputs': inputs,
                                          'input_verification': 'verified' if not failures and primary == 'verified'
                                                                else 'incomplete',
                                          'acquisition_evidence': proof}
                                if failures:
                                    status = failures[0]['status']
                                    atomic_json(state_path, {**common, 'status': status, 'failed_units': failures})
                                    record(label, status)
                                    if any(x.get('retryable', True) for x in failures):
                                        pending.add((lo, hi))
                                    progress.finish(counts)
                                    continue
                                if primary != 'verified':
                                    atomic_json(state_path, {**common, 'status': 'remote_absent',
                                                            'reason': 'Primary daily granule absent in verified remote index'})
                                    record(label, 'remote_absent')
                                    progress.finish(counts)
                                    continue
                                if resume_valid(state_path, output, identity, signature):
                                    record(label, 'skipped_verified')
                                    progress.finish(counts)
                                    continue
                                progress.current(lo, 'wait/stage')
                                wait_resources(args, scratch)
                                check_sources(config)
                                try:
                                    for source in particles + mag:
                                        if (str(source) not in staged
                                                or staged[str(source)]['source'] != fingerprint(source)):
                                            date_match = re.search(r'_(\d{8})(?:\d{2})?_v', source.name)
                                            source_cache = scratch / 'cdf' / date_match.group(1)
                                            staged[str(source)] = stage_cdf(source, source_cache, args)
                                    if [staged[str(p)]['source'] for p in particles + mag] != inputs:
                                        raise ValueError('Input changed while staging; restart to use a consistent snapshot')
                                except (OSError, ValueError, EOFError, MemoryError) as exc:
                                    atomic_json(state_path, {**common, 'status': 'failed_stage',
                                                            'reason': str(exc)})
                                    record(label, 'failed_stage')
                                    progress.finish(counts)
                                    print(f'{lo.isoformat()} failed_stage: {exc}', flush=True)
                                    # Do not attempt the same broken day-CDF hundreds of times.
                                    break
                                request = {**common, 'config': config, 'output': str(output),
                                           'lepi': [staged[str(p)] for p in particles],
                                           'mgf': [staged[str(p)] for p in mag]}
                                request_path = cache / 'request.json'
                                result_path = cache / 'result.json'
                                atomic_json(request_path, request)
                                result_path.unlink(missing_ok=True)
                                log = month_dir / 'logs' / day_label / (label + '.log')
                                log.parent.mkdir(parents=True, exist_ok=True)
                                atomic_json(state_path, {**common, 'status': 'running'})
                                check_sources(config)
                                if output.exists():
                                    # Preserve an incompatible/corrupt old artifact for audit.
                                    obsolete = month_dir / 'superseded' / day_label
                                    obsolete.mkdir(parents=True, exist_ok=True)
                                    old_name = output.stem + '_' + sha_file(output)[:12] + '.npz'
                                    os.replace(output, obsolete / old_name)
                                cmd = [sys.executable, str(BASE / 'worker.py'),
                                       '--request', str(request_path), '--result', str(result_path)]
                                # Child uses its own fresh process group so timeout/interrupt can clean it up.
                                progress.current(lo, 'moments')
                                proc = None
                                try:
                                    with log.open('w') as stream:
                                        proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                                                start_new_session=True)
                                        try:
                                            code = proc.wait(timeout=args.worker_timeout)
                                        except BaseException:
                                            os.killpg(proc.pid, 15)
                                            try:
                                                proc.wait(timeout=5)
                                            except subprocess.TimeoutExpired:
                                                os.killpg(proc.pid, 9)
                                                proc.wait()
                                            raise
                                    if code != 0 or not result_path.exists():
                                        state = {**common, 'status': 'failed', 'returncode': code, 'log': str(log)}
                                    else:
                                        state = {**common, **json.loads(result_path.read_text()), 'log': str(log)}
                                except subprocess.TimeoutExpired:
                                    state = {**common, 'status': 'timeout', 'log': str(log)}
                                except KeyboardInterrupt:
                                    atomic_json(state_path, {**common, 'status': 'interrupted', 'log': str(log)})
                                    raise
                                atomic_json(state_path, state)
                                record(label, state['status'])
                                progress.finish(counts)
                                print(f'{lo.isoformat()} {state["status"]}', flush=True)
                                time.sleep(args.pause_seconds)
                        finally:
                            # Only this profile's explicitly owned, day-specific scratch is removed.
                            if cache.exists():
                                shutil.rmtree(cache)
                            # Keep today for tomorrow's boundary, and any lookahead day.
                            # Earlier granules have now served their final boundary use.
                            for old in (scratch / 'cdf').glob('*'):
                                if old.is_dir() and re.fullmatch(r'\d{8}', old.name) and old.name < day_label:
                                    shutil.rmtree(old)
                            monthly_summary(month_dir)
                        print(json.dumps({'day': day_label, 'request_counts_so_far': dict(counts)}), flush=True)
                report = {'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
                          'profile_hash': identity, 'output_root': str(root), 'counts': dict(counts)}
                atomic_json(root / 'last_run.json', report)
                if (scratch / 'cdf').exists():
                    shutil.rmtree(scratch / 'cdf')
    print(json.dumps(report), flush=True)
    incomplete = sum(v for k, v in counts.items() if k not in {'complete', 'no_data', 'remote_absent', 'skipped_verified'})
    return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--month')
    selection.add_argument('--year', type=int)
    selection.add_argument('--start', help='UTC or explicit offset; requires --end')
    parser.add_argument('--end')
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scratch-root', type=Path, default=Path('/tmp/lepi_moments_work'))
    parser.add_argument('--mgf-cadence', choices=['64hz', '8sec'], default='64hz')
    parser.add_argument('--max-mgf-gap-seconds', type=float,
                        help='Default 1 s for 64 Hz, 16 s for 8 s data; no interpolation beyond this gap')
    parser.add_argument('--min-free-memory-gib', type=float, default=8.)
    parser.add_argument('--scratch-reserve-gib', type=float, default=4.)
    parser.add_argument('--pause-seconds', type=float, default=2.)
    parser.add_argument('--worker-timeout', type=float, default=180.)
    parser.add_argument('--offline', action='store_true', help='Use previously verified shared cache only')
    parser.add_argument('--refresh-downloads', action='store_true', help='Recheck remote versions and inputs')
    parser.add_argument('--remote-root', default='https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/')
    parser.add_argument('--download-attempts', type=int, default=3)
    parser.add_argument('--lepi-boundary-tolerance-seconds', type=float, default=64.,
                        help='Daily filename guard for native integration-start epochs; does not shift timestamps')
    parser.add_argument('--max-network-failures', type=int, default=0,
                        help='Optional circuit breaker; 0 continues and defers failed units (default)')
    parser.add_argument('--retry-passes', type=int, default=1,
                        help='Retry failed acquisition chunks after the complete first calculation pass')
    parser.add_argument('--retry-pass-delay', type=float, default=30.)
    parser.add_argument('--retry-delay', type=float, default=30.)
    parser.add_argument('--http-timeout', type=float, default=60.)
    parser.add_argument('--download-timeout', type=float, default=600.)
    parser.add_argument('--validation-timeout', type=float, default=180.)
    parser.add_argument('--progress', choices=['auto', 'on', 'off'], default='auto',
                        help='Progress bar on stderr: auto enables it only in a terminal')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.max_mgf_gap_seconds is None:
        args.max_mgf_gap_seconds = 1. if args.mgf_cadence == '64hz' else 16.
    limits = [args.max_mgf_gap_seconds, args.worker_timeout, args.min_free_memory_gib,
              args.scratch_reserve_gib, args.pause_seconds, args.retry_delay, args.http_timeout,
              args.download_timeout, args.validation_timeout, args.lepi_boundary_tolerance_seconds, args.retry_pass_delay]
    if not all(math.isfinite(x) for x in limits) or min(args.max_mgf_gap_seconds, args.worker_timeout) <= 0 or min(
            args.min_free_memory_gib, args.scratch_reserve_gib, args.pause_seconds) < 0:
        parser.error('Invalid resource or interpolation limit')
    if (args.download_attempts < 1 or args.max_network_failures < 0 or args.retry_passes < 0 or min(args.http_timeout, args.download_timeout, args.validation_timeout) <= 0
            or min(args.retry_delay, args.lepi_boundary_tolerance_seconds, args.retry_pass_delay) < 0 or (args.offline and args.refresh_downloads)):
        parser.error('Invalid acquisition options')
    try:
        start, end = requested_range(args)
        identity, config = profile(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        inventory = Counter()
        for day, lo, hi in days(start, end):
            inventory['days'] += 1
            inventory['planned_chunks'] += sum(1 for _ in chunks(lo, hi))
            inventory['days_with_local_lepi'] += int(source_day(args.input_root, day) is not None)
        print(json.dumps({'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
                          'profile_hash': identity, 'inventory': dict(inventory),
                          'output': str(args.output_root / ('native_dsi_' + identity[:12]))}, indent=2))
        return 0
    def stop_requested(signum, frame):
        raise KeyboardInterrupt(f'Stop signal {signum}')
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    # Idle I/O priority inherited by the worker, when ionice is available.
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    try:
        return run(args, start, end, identity, config)
    except BlockingIOError:
        print('Another runner holds the output/scratch lock; no worker started', file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print('Interrupted; restart the same command to resume', file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'Runner stopped: {exc}; restart the same command after resolving the cause', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
