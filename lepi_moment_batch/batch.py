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
import zipfile
import io

BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
DEFAULT_INPUT = Path(os.environ.get('SPEDAS_DATA_DIR', '/mnt/j/observation_data')) / 'ergsc/satellite/erg'
DEFAULT_OUTPUT = Path('/mnt/j/observation_data/Arase_LEPi_moments')
UTC = dt.timezone.utc
CHUNK_SECONDS = 300
MOMENT_TIME_POLICY = 'per_particle_mgf_valid_tt2000_v2'
MOMENT_INVALID_BITS = {'outside_mgf_support': 1, 'nonfinite_mgf_bracket': 2,
                       'mgf_gap_exceeded': 4, 'zero_interpolated_mgf': 8,
                       'conflicting_mgf_epoch': 16, 'quarantined_mgf_interval': 32}


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
    if getattr(args, 'recovery_plan', None):
        from recovery_plan import load, ranges
        selected = ranges(load(args.recovery_plan))
        return selected[0][0], selected[-1][1]
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
             BASE / 'recovery_plan.py', BASE / 'mgf_policy.py', BASE / 'validate_input.py', BASE / 'acquisition_worker.py', BASE / 'compute.py', WORKSPACE / 'lepi_moment_pilot/pilot.py']
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
              'duplicate_record_policy': 'LEPi: verified identical within granule, last dated source across granules; MGF: identical only, conflicting epochs block interpolation',
              'sub_anodes_used': False, 'mgf_cadence': args.mgf_cadence,
              'max_mgf_gap_seconds': args.max_mgf_gap_seconds,
              'moment_time_policy': MOMENT_TIME_POLICY,
              'mgf_record_policy': 'mgf_record_quarantine_tt2000_v1',
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


def date_key(source):
    return re.search(r'_(\d{8})(?:\d{2})?_v', Path(source).name).group(1)


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
        if state['status'] == 'excluded':
            return (state.get('classification_version') in (1, 2) and state.get('retryable') is False
                    and state.get('exclusion_basis') in {'verified_remote_index', 'verified_cdf_timestamps', 'verified_mgf_coverage'}
                    and not output.exists())
        return (state['status'] in {'complete', 'partial'} and state.get('input_verification') == 'verified' and output.exists()
                and state['output_sha256'] == sha_file(output))
    except (OSError, ValueError, KeyError):
        return False


# Only these reviewed predecessor implementations may supply reusable results.
REUSABLE_RUNNER_SOURCES = {
    str(BASE / 'batch.py'): '472276f697d77f7e5b6aea47616769bc5d701ea82b8671159d0867c54408f4c6',
    str(BASE / 'worker.py'): 'c41d4d91211b9d7f57623c4f1bae5184e54788fd6007d0b34d10bd03324e338c',
    str(BASE / 'acquisition.py'): '755fb58afb0336903d6b3936d83cbb56e4194e7a6704186ec12286898daef14f',
}
REUSABLE_PARALLEL_DOWNLOAD_SOURCES = {
    str(BASE / 'batch.py'): '82d571609b75dd450f9b2af3c1e6f52f528c8cce7326565325cbb68a51c1bb62',
    str(BASE / 'acquisition.py'): '738d42352aac4b827f80e49a208e707188f2c09c0861b9e7fa68ddf31b38a1ec',
}
REUSABLE_COMPUTE_SOURCES = {
    str(BASE / 'acquisition.py'): '22c8bf81888f25d30879fe5e0114c8abef54991f17a17801c72cfdb8a2b45577',
    str(BASE / 'batch.py'): 'c1edda83f4a03a141d51d1e290fff29529fce17c25ce98f24be3d1dad8d1fa0b',
    str(BASE / 'worker.py'): '5ae0ac72bb4522177859a632b3db4de51cca6308c6e5f007d419271c3da5bc78',
    str(BASE / 'compute.py'): 'a204ca942d52e01d0006e199f25cbd35520568107fc23ff92fa1223cf13a889f',
}
REUSABLE_FAST_RUNNER_SOURCES = {
    str(BASE / 'batch.py'): '334992358701fbb277e7fdc13bf7c26da5d1037c82e291778b2c61d9c6013af8',
    str(BASE / 'worker.py'): '5108d90b2e8c933c4415ece5fcdfb403caf2a1f8e6a5be67dfdacf20e43e2101',
}


def compatible_reuse_root(root, config):
    root = Path(root).resolve()
    profile = json.loads((root / 'profile.json').read_text())
    old = profile['config']
    identity = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    if identity != profile['profile_hash'] or root.name != 'native_dsi_' + identity[:12]:
        raise ValueError('Reuse profile identity is invalid')
    if (old.get('moment_time_policy') not in (None, config.get('moment_time_policy'))
            or {k: v for k, v in old.items() if k not in {'sources', 'moment_time_policy'}}
            != {k: v for k, v in config.items() if k not in {'sources', 'moment_time_policy'}}):
        raise ValueError('Reuse calculation conditions/packages differ')
    added = {str(BASE / 'acquisition_worker.py'), str(BASE / 'compute.py')}
    if set(old['sources']) - set(config['sources']) or set(config['sources']) - set(old['sources']) - added:
        raise ValueError('Reuse source inventory differs')
    for path, old_hash in old['sources'].items():
        if old_hash not in (config['sources'][path], REUSABLE_RUNNER_SOURCES.get(path),
                            REUSABLE_FAST_RUNNER_SOURCES.get(path), REUSABLE_PARALLEL_DOWNLOAD_SOURCES.get(path),
                            REUSABLE_COMPUTE_SOURCES.get(path)):
            raise ValueError('Reuse scientific/unsupported source differs: ' + path)
    return root, identity


def reuse_result(previous, root, state_path, output, identity, signature, common, config):
    if previous is None or output.exists():
        return False
    old_root, old_identity = previous
    old_state = old_root / state_path.relative_to(root)
    old_output = old_root / output.relative_to(root)
    try:
        state = json.loads(old_state.read_text())
        if state.get('status') != 'complete' or not resume_valid(old_state, old_output, old_identity, state['input_signature']):
            return False
        # A validator identity can change after orchestration edits; the verified
        # raw bytes, file fingerprints, units and absence evidence must still match.
        def evidence(proof):
            return [{k: v for k, v in x.items() if k != 'validation_identity'} for x in proof]
        if (common['input_verification'] != 'verified'
                or evidence(state['acquisition_evidence']) != evidence(common['acquisition_evidence'])):
            return False
        if state['status'] == 'complete':
            output.parent.mkdir(parents=True, exist_ok=True)
            # Preserve every numerical .npy payload byte-for-byte. Only the
            # manifest is updated so downstream GSM conversion sees this profile.
            import numpy as np
            fd, temp = tempfile.mkstemp(dir=output.parent, suffix='.reuse')
            try:
                with os.fdopen(fd, 'wb') as dest, zipfile.ZipFile(old_output) as source:
                    manifest = json.loads(str(np.load(io.BytesIO(source.read('manifest_json.npy')), allow_pickle=False)))
                    if manifest['profile_hash'] != old_identity:
                        return False
                    epoch = np.load(io.BytesIO(source.read('epoch_tt2000.npy')), allow_pickle=False)
                    n = len(epoch)
                    added_arrays = {}
                    for name, values in {'moment_valid': np.ones(n, dtype=bool),
                                         'moment_invalid_reason': np.zeros(n, dtype=np.uint8)}.items():
                        if name + '.npy' not in source.namelist():
                            added_arrays[name] = values
                        elif not np.array_equal(np.load(io.BytesIO(source.read(name+'.npy')), allow_pickle=False), values):
                            return False
                    metadata = manifest.setdefault('metadata', {})
                    metadata['moment_selection'] = {'policy': MOMENT_TIME_POLICY, 'invalid_reason_bits': MOMENT_INVALID_BITS,
                                                    'records': n, 'computed_records': n, 'invalid_records': 0,
                                                    'invalid_reason_counts': {key: 0 for key in MOMENT_INVALID_BITS},
                                                    'legacy_complete_upgraded': True}
                    for name, value in added_arrays.items():
                        manifest.setdefault('array_sha256', {})[name] = hashlib.sha256(value.tobytes()).hexdigest()
                    manifest.update(profile_hash=identity, input_signature=signature, config=config,
                                    acquisition_evidence=common['acquisition_evidence'],
                                    reused_from={'profile_hash': old_identity, 'output': str(old_output),
                                                 'output_sha256': state['output_sha256']})
                    buffer = io.BytesIO()
                    np.save(buffer, json.dumps(manifest), allow_pickle=False)
                    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as target:
                        for entry in source.infolist():
                            if entry.filename == 'manifest_json.npy':
                                target.writestr(entry.filename, buffer.getvalue())
                            else:
                                with source.open(entry) as inp, target.open(entry.filename, 'w', force_zip64=True) as out:
                                    shutil.copyfileobj(inp, out, 1024**2)
                        for name, value in added_arrays.items():
                            array_buffer = io.BytesIO()
                            np.save(array_buffer, value, allow_pickle=False)
                            target.writestr(name+'.npy', array_buffer.getvalue())
                    dest.flush()
                    os.fsync(dest.fileno())
                if sha_file(old_output) != state['output_sha256']:
                    return False
                os.replace(temp, output)
                state.update(output=str(output), output_sha256=sha_file(output), file_bytes=output.stat().st_size,
                             computed_records_per_species=n, invalid_records_per_species=0, valid_moment_fraction=1.,
                             moment_invalid_reason_counts={key: 0 for key in MOMENT_INVALID_BITS})
            finally:
                Path(temp).unlink(missing_ok=True)
        elif output.exists():
            return False
        atomic_json(state_path, {**state, **common, 'profile_hash': identity,
                                'input_signature': signature,
                                'reused_from': {'profile_hash': old_identity, 'state': str(old_state),
                                                'output': str(old_output) if state['status'] == 'complete' else None}})
        return True
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
        return False


def reuse_profiles(paths, config):
    """Validate candidate profiles once, preserving the user's priority order."""
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    candidates = []
    seen = set()
    for path in paths:
        candidate = compatible_reuse_root(path, config)
        if candidate[0] not in seen:
            candidates.append(candidate)
            seen.add(candidate[0])
    return candidates


def reuse_first_result(candidates, *args):
    """Try the next profile when a chunk is absent, failed or fails integrity checks."""
    return any(reuse_result(candidate, *args) for candidate in candidates)


def monthly_summary(month_dir):
    statuses = Counter()
    reasons = Counter()
    record_counts = Counter()
    for path in (month_dir / 'state').glob('*/*.json'):
        try:
            state = json.loads(path.read_text())
            statuses[state['status']] += 1
            if state['status'] in {'complete', 'partial'}:
                n = state.get('records_per_species', 0)
                record_counts['native_records_in_saved_products'] += n
                record_counts['computed_records_in_saved_products'] += state.get('computed_records_per_species', n)
                record_counts['invalid_records_in_saved_products'] += state.get('invalid_records_per_species', 0)
            if state.get('failure_class'):
                reasons[state['failure_class']] += 1
        except (ValueError, KeyError):
            statuses['invalid_state'] += 1
    # This is an inventory, not a claim of full-month observation coverage.
    atomic_json(month_dir / 'summary.json', {'updated_utc': dt.datetime.now(UTC).isoformat(),
                'chunk_state_counts': dict(statuses), 'failure_class_counts': dict(reasons),
                'record_counts': dict(record_counts),
                'scope': 'Inventory of attempted chunks; does not imply a fully processed month'})


@contextmanager
def track_run(root, start, end, identity, counts, selection_metadata=None):
    base = {'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
            'profile_hash': identity, 'output_root': str(root), **(selection_metadata or {})}
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


def acquisition_failure_class(failures):
    item = failures[0]
    status = item['status']
    if status == 'failed_authorization':
        return 'authorization_error'
    if status == 'failed_validation':
        return 'input_validation_error'
    attempts = item.get('attempts', [])
    timed_out = any('timeout' in x.get('reason', '').lower() or 'timed out' in x.get('reason', '').lower() for x in attempts)
    if timed_out:
        return 'catalog_timeout' if attempts[-1].get('phase') == 'catalog' else 'download_timeout'
    if attempts and attempts[-1].get('http_status'):
        return 'http_error'
    return 'download_error' if attempts else 'unverified_input'


def run(args, start, end, identity, config):
    root = args.output_root / ('native_dsi_' + identity[:12])
    reuse_candidates = reuse_profiles(getattr(args, 'reuse_results_from', None), config)
    root.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch_root / identity[:12]
    scratch.mkdir(parents=True, exist_ok=True)
    from compute import ComputePool
    from acquisition import Acquisition
    acquisition = Acquisition(args, scratch)
    from progress import Progress
    initial_selection = None
    if getattr(args, 'recovery_plan', None):
        from recovery_plan import load, ranges, CONDITIONS
        plan = load(args.recovery_plan)
        if not args.offline:
            raise ValueError('Recovery plans require --offline')
        if {k:config[k] for k in CONDITIONS} != plan['conditions']:
            raise ValueError('Recovery calculation conditions differ from the source profile')
        initial_selection = set(ranges(plan))
    total = len(initial_selection) if initial_selection is not None else int((end - start).total_seconds()) // CHUNK_SECONDS
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
    def finished(label, state):
        record(label, state['status'])
        progress.finish(counts)
        print(f'{state["start_utc"]} {state["status"]}: {state.get("failure_class", "")}', flush=True)

    with Progress(total, getattr(args, 'progress', 'auto')) as progress, (root / '.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A separate lock protects scratch even if an alternate output root is used.
        with (scratch / '.scratch.lock').open('a+') as scratch_lock:
            fcntl.flock(scratch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            selection_metadata = ({'selected_chunks':len(initial_selection), 'recovery_plan_sha256':plan['plan_sha256']}
                                  if initial_selection is not None else None)
            with track_run(root, start, end, identity, counts, selection_metadata):
                atomic_json(root / 'profile.json', {'profile_hash': identity, 'config': config})
                if getattr(args, 'recovery_plan', None):
                    atomic_json(root / 'recovery_plan.json', plan)
                selected = initial_selection
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
                            with ComputePool(args, finished) as compute:
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
                                        reason = acquisition_failure_class(failures)
                                        atomic_json(state_path, {**common, 'status': status, 'failure_class': reason,
                                                                'reason': failures[0].get('reason', ''),
                                                                'retryable': any(x.get('retryable', True) for x in failures),
                                                                'failed_units': failures})
                                        record(label, status)
                                        if any(x.get('retryable', True) for x in failures):
                                            pending.add((lo, hi))
                                        progress.finish(counts)
                                        continue
                                    if primary != 'verified':
                                        atomic_json(state_path, {**common, 'status': 'excluded',
                                                                'failure_class': 'required_input_missing_remote',
                                                                'classification_version': 1, 'retryable': False,
                                                                'exclusion_basis': 'verified_remote_index',
                                                                'reason': 'Primary daily granule absent in verified remote index'})
                                        record(label, 'excluded')
                                        progress.finish(counts)
                                        continue
                                    try:
                                        excluded_before = json.loads(state_path.read_text()).get('status') == 'excluded'
                                    except (OSError, ValueError):
                                        excluded_before = False
                                    if ((not (excluded_before and getattr(args, 'recheck_excluded', False))
                                         and resume_valid(state_path, output, identity, signature))
                                            or reuse_first_result(reuse_candidates, root, state_path, output, identity, signature, common, config)):
                                        record(label, ('skipped_excluded' if excluded_before else
                                                       'skipped_partial' if json.loads(state_path.read_text())['status'] == 'partial'
                                                       else 'skipped_verified'))
                                        progress.finish(counts)
                                        continue
                                    compute.wait_for_slot()
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
                                        for source in particles + mag:
                                            item = staged[str(source)]
                                            key = hashlib.sha256(json.dumps(item['source'], sort_keys=True).encode()).hexdigest()
                                            item['time_cache_path'] = str(scratch / 'cdf' / date_key(source) / (key + '.epochs.npz'))
                                        if [staged[str(p)]['source'] for p in particles + mag] != inputs:
                                            raise ValueError('Input changed while staging; restart to use a consistent snapshot')
                                    except (OSError, ValueError, EOFError, MemoryError) as exc:
                                        atomic_json(state_path, {**common, 'status': 'failed_stage',
                                                                'failure_class': 'input_stage_error', 'retryable': True,
                                                                'reason': str(exc)})
                                        record(label, 'failed_stage')
                                        progress.finish(counts)
                                        print(f'{lo.isoformat()} failed_stage: {exc}', flush=True)
                                        # Do not attempt the same broken day-CDF hundreds of times.
                                        break
                                    request = {**common, 'config': config, 'output': str(output),
                                               'lepi': [staged[str(p)] for p in particles],
                                               'mgf': [staged[str(p)] for p in mag]}
                                    if output.exists():
                                        # Preserve an incompatible/corrupt old artifact for audit.
                                        obsolete = month_dir / 'superseded' / day_label
                                        obsolete.mkdir(parents=True, exist_ok=True)
                                        old_name = output.stem + '_' + sha_file(output)[:12] + '.npz'
                                        os.replace(output, obsolete / old_name)
                                    log = month_dir / 'logs' / day_label / (label + '.log')
                                    compute.submit(label, request, common, state_path, cache / label, log)
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
                if initial_selection is not None:
                    report.update(selected_chunks=len(initial_selection), recovery_plan_sha256=plan['plan_sha256'])
                atomic_json(root / 'last_run.json', report)
                if (scratch / 'cdf').exists():
                    shutil.rmtree(scratch / 'cdf')
    print(json.dumps(report), flush=True)
    incomplete = sum(v for k, v in counts.items() if k not in {'complete', 'partial', 'skipped_partial', 'excluded', 'skipped_excluded', 'skipped_verified'})
    return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--month')
    selection.add_argument('--year', type=int)
    selection.add_argument('--recovery-plan', type=Path, help='Frozen five-minute selection and local inputs; requires --offline')
    selection.add_argument('--start', help='UTC or explicit offset; requires --end')
    parser.add_argument('--end')
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--reuse-results-from', type=Path, action='append',
                        help='Repeat to search compatible profiles in order; only verified complete results are reused')
    parser.add_argument('--scratch-root', type=Path, default=Path('/tmp/lepi_moments_work'))
    parser.add_argument('--mgf-cadence', choices=['64hz', '8sec'], default='64hz')
    parser.add_argument('--max-mgf-gap-seconds', type=float,
                        help='Default 1 s for 64 Hz, 16 s for 8 s data; no interpolation beyond this gap')
    parser.add_argument('--min-free-memory-gib', type=float, default=8.)
    parser.add_argument('--scratch-reserve-gib', type=float, default=4.)
    parser.add_argument('--pause-seconds', type=float, default=2.)
    parser.add_argument('--compute-workers', type=int, default=1,
                        help='Concurrent 5-minute compute processes, one distinct CPU each (default: 1)')
    parser.add_argument('--recheck-excluded', action='store_true',
                        help='Reevaluate excluded chunks in the current profile')
    parser.add_argument('--worker-timeout', type=float, default=180.)
    parser.add_argument('--offline', action='store_true', help='Use previously verified shared cache only')
    parser.add_argument('--refresh-downloads', action='store_true', help='Recheck remote versions and inputs')
    parser.add_argument('--remote-root', default='https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/')
    parser.add_argument('--download-attempts', type=int, default=3)
    parser.add_argument('--download-workers', type=int, default=8,
                        help='Parallel prefetch acquisition processes (default: 8; validation stays serial)')
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
    if (args.compute_workers < 1 or args.compute_workers > len(os.sched_getaffinity(0))
            or args.download_workers < 1 or args.download_attempts < 1 or args.max_network_failures < 0 or args.retry_passes < 0 or min(args.http_timeout, args.download_timeout, args.validation_timeout) <= 0
            or min(args.retry_delay, args.lepi_boundary_tolerance_seconds, args.retry_pass_delay) < 0 or (args.offline and args.refresh_downloads)):
        parser.error('Invalid acquisition options')
    if args.recovery_plan and (not args.offline or args.end):
        parser.error('--recovery-plan requires --offline and cannot be combined with --end')
    try:
        start, end = requested_range(args)
        identity, config = profile(args)
        if args.recovery_plan:
            from recovery_plan import load, CONDITIONS
            if {k:config[k] for k in CONDITIONS} != load(args.recovery_plan)['conditions']:
                raise ValueError('Recovery calculation conditions differ from the source profile')
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        inventory = Counter()
        if args.recovery_plan:
            from recovery_plan import load, ranges
            selected = ranges(load(args.recovery_plan))
            inventory['selected_chunks'] = len(selected)
            inventory['local_input_files'] = sum(x['status']=='local_candidate' for x in load(args.recovery_plan)['inputs'].values())
            inventory['proven_absent_units'] = sum(x['status']=='remote_absent' for x in load(args.recovery_plan)['inputs'].values())
        for day, lo, hi in days(start, end):
            if args.recovery_plan and not any(lo <= a < hi for a,_ in selected):
                continue
            inventory['days'] += 1
            inventory['planned_chunks'] += sum(lo <= a < hi for a,_ in selected) if args.recovery_plan else sum(1 for _ in chunks(lo, hi))
            inventory['days_with_local_lepi'] += int(source_day(args.input_root, day) is not None)
        print(json.dumps({'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
                          'profile_hash': identity, 'inventory': dict(inventory),
                          'output': str(args.output_root / ('native_dsi_' + identity[:12]))}, indent=2))
        return 0
    def stop_requested(signum, frame):
        raise KeyboardInterrupt(f'Stop signal {signum}')
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    args.compute_cpu_ids = sorted(os.sched_getaffinity(0))[-args.compute_workers:]
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
