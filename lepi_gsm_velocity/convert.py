"""Separate resumable GSM observed-velocity products from completed native DSI moments."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
# Set BEFORE importing pySPEDAS. Existing DSI batch sources are never modified/imported.
os.environ.setdefault('SPEDAS_DATA_DIR', '/mnt/j/observation_data/')
os.environ.setdefault('SPACEPY', '/tmp/lepi_gsm_spacepy')
Path(os.environ['SPACEPY']).mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lepi_gsm_matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from core import FLAGS, apply, rotation, series
UTC = dt.timezone.utc
SPECIES = ('FPDU', 'FHEDU', 'FODU')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fp(path):
    p = Path(path).resolve(); st = p.stat()
    return {'path': str(p), 'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def save(path, arrays, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays, manifest_json=json.dumps(manifest)); f.flush(); os.fsync(f.fileno())
        with np.load(temp, allow_pickle=False) as check:
            for key, value in arrays.items():
                np.testing.assert_array_equal(check[key], value)
        assert_snapshot(manifest['inputs'])
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def assert_snapshot(inputs):
    for item in inputs:
        if fp(item['fingerprint']['path']) != item['fingerprint']:
            raise RuntimeError('Input changed during conversion: '+item['fingerprint']['path'])


def profile(args, source_profile):
    import pyspedas
    from pyspedas.projects.erg.satellite.erg.common.cotrans import dsi2j2000, cart_trans_matrix_make
    from pyspedas.cotrans_tools import cotrans, cotrans_lib
    from pyspedas.projects.erg.satellite.erg.att.att import att
    from pyspedas.projects.erg.satellite.erg.orb.orb import orb
    import inspect
    modules = (dsi2j2000, cart_trans_matrix_make, cotrans, cotrans_lib, att, orb)
    paths = [BASE/'convert.py', BASE/'core.py'] + [Path(inspect.getfile(m)) for m in modules]
    paths += sorted(Path(inspect.getfile(cotrans)).parent.glob('*.py'))
    config = {'schema': 'lepi_gsm_observed_velocity_v1', 'source_profile_hash': source_profile['profile_hash'],
              'source_moments_root': str(args.moments_root.resolve()), 'units': 'km/s',
              'frame': 'GSM axes; spacecraft-relative observed velocity; no spacecraft-velocity correction',
              'path': 'DSI -> J2000 -> GSM; public pySPEDAS basis with strict support interpolation',
              'species': list(SPECIES), 'attitude_gap_seconds': args.max_attitude_gap,
              'orbit_gap_seconds': args.max_orbit_gap, 'nominal_sun_distance_km': 1.496e8,
              'attitude_quality_mask_applied': False, 'attitude_row_layouts': '16 fields; 14 fields with omitted FLAG_VI and Sensor', 'orbit_product': 'L2 definitive pos_gse; CDF UNITS checked, RE converted to km',
              'earth_radius_km': args.earth_radius_km,
              'extrapolation': False, 'sources': {str(p): sha(p) for p in paths},
              'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'cdflib', 'pyspedas')}}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return identity, config


def check_code(config):
    for path, expected in config['sources'].items():
        if sha(path) != expected:
            raise RuntimeError('Conversion source changed during run: '+path)
    for name, expected in config['packages'].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError('Package changed during run: '+name)


def parse(value):
    d = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)


def selected_range(args):
    if args.month:
        if not re.fullmatch(r'\d{4}-\d{2}', args.month): raise ValueError('--month must be YYYY-MM')
        start = parse(args.month+'-01')
        end = (start.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
    elif args.year is not None:
        if not 1 <= args.year <= 9998: raise ValueError('--year must be 1..9998')
        start = dt.datetime(args.year, 1, 1, tzinfo=UTC); end = start.replace(year=start.year+1)
    else:
        if not args.end:
            raise ValueError('--start requires --end')
        start, end = parse(args.start), parse(args.end)
    if end <= start or (args.end and not args.start):
        raise ValueError('Invalid time selection')
    return start, end


def inventory(root, start, end):
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    found = []
    while cursor < end:
        for path in (root/cursor.strftime('%Y/%m')/'state').glob('*/*.json'):
            state = json.loads(path.read_text())
            lo, hi = parse(state['start_utc']), parse(state['end_utc_exclusive'])
            if start <= lo and hi <= end and state['status'] in {'complete', 'partial'}:
                found.append((lo, path))
        cursor = (cursor.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
    return sorted(found)


def available_gib():
    return next(int(line.split()[1])/1024**2 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def attitude_quality_rows(content):
    if 'J2000' not in content:
        raise ValueError('Attitude file does not declare J2000 axes')
    times, rows = [], []
    for line in content.splitlines():
        columns = line.split()
        if not columns or columns[0].startswith('#'):
            continue
        if len(columns) == 16:
            quality = columns[7:9]+columns[14:16]
        elif len(columns) == 14:
            # Some real records omit FLAG_VI and Sensor. Other columns shift left by two.
            float(columns[7])  # require the numeric Phase field at this position
            quality = ['', '']+columns[12:14]
        else:
            raise ValueError('Unsupported attitude row layout: '+str(len(columns)))
        float(quality[2]); float(quality[3])
        times.append(columns[0]); rows.append(quality)
    if not rows:
        raise ValueError('Empty attitude ASCII data')
    return times, rows


class Support:
    def __init__(self, args):
        self.args = args
        self.day = None

    def load(self, day):
        import pyspedas as psp
        from pyspedas.projects.erg.satellite.erg.att.att import att
        from pyspedas.projects.erg.satellite.erg.orb.orb import orb
        # Full day plus boundary brackets, loaded sequentially into this process only.
        trange = [(day-dt.timedelta(seconds=120)).strftime('%Y-%m-%d/%H:%M:%S'),
                  (day+dt.timedelta(days=1, seconds=120)).strftime('%Y-%m-%d/%H:%M:%S')]
        attempts = 1 if self.args.offline else self.args.download_attempts
        errors = []
        for number in range(attempts):
            psp.del_data('erg_att_*'); psp.del_data('erg_orb_l2_*')
            try:
                force = number > 0 and not self.args.offline
                af = att(trange=trange, downloadonly=True, no_update=self.args.offline, force_download=force)
                of = orb(trange=trange, level='l2', datatype='def', downloadonly=True,
                         no_update=self.args.offline, force_download=force, ror=False)
                if not af or not of:
                    raise RuntimeError('Required attitude/orbit files not acquired; absence unconfirmed')
                inputs = [{'fingerprint': fp(p), 'sha256': sha(p)} for p in sorted(set(af+of))]
                att(trange=trange, no_update=True)
                orb(trange=trange, level='l2', datatype='def', no_update=True, ror=False)
                ra = psp.get_data('erg_att_izras'); dec = psp.get_data('erg_att_izdec')
                pos = psp.get_data('erg_orb_l2_pos_gse')
                if ra is None or dec is None or pos is None:
                    raise RuntimeError('Required tplot support variables missing after load')
                np.testing.assert_array_equal(ra.times, dec.times)
                at, angles = series(ra.times, np.column_stack((ra.y, dec.y)))
                ot, positions = series(pos.times, pos.y)
                # Load original attitude quality, which the public tplot loader omits.
                from pyspedas.tplot_tools import time_float
                raw_times, raw_quality = [], []
                for name in af:
                    rt, rq = attitude_quality_rows(Path(name).read_text())
                    raw_times.extend(rt); raw_quality.extend(rq)
                qt, quality = series(np.asarray(time_float(raw_times)), np.asarray(raw_quality))
                if qt.shape != at.shape: raise ValueError('Attitude quality record count mismatch')
                np.testing.assert_allclose(qt, at, rtol=0., atol=1.e-6)
                # Required pos_gse units must be km; keep CDF evidence in the manifest.
                import cdflib
                orbit_units = set()
                for name in of:
                    with cdflib.CDF(name) as cdf:
                        units = str(cdf.varattsget('pos_gse').get('UNITS', '')).strip().lower()
                        if units not in ('km', 're'):
                            raise ValueError('Unsupported pos_gse units: '+units)
                        orbit_units.add(units)
                        cdf.varget('pos_gse')  # read all required records, including the tail
                if len(orbit_units) != 1:
                    raise ValueError('Mixed orbit position units')
                if orbit_units == {'re'}:
                    positions = positions*self.args.earth_radius_km
                for item in inputs:
                    item['role'] = 'attitude_ascii' if item['fingerprint']['path'].endswith('.txt') else 'orbit_cdf'
                self.orbit_input_units = next(iter(orbit_units))
                assert_snapshot(inputs)
                self.day, self.at, self.angles, self.ot, self.positions = day, at, angles, ot, positions
                self.quality, self.inputs = quality, inputs
                return
            except Exception as exc:
                errors.append(str(exc))
                if number+1 < attempts:
                    time.sleep(self.args.retry_delay)
        raise RuntimeError('Support acquisition/load failed: '+'; '.join(errors))

    def ensure(self, day):
        if self.day != day:
            self.load(day)
        try:
            assert_snapshot(self.inputs)
        except RuntimeError:
            self.load(day)


def convert_chunk(state_path, source_root, root, identity, config, support):
    state = json.loads(state_path.read_text())
    if state['status'] not in {'complete', 'partial'} or state.get('input_verification') != 'verified':
        raise ValueError('Source moment state is not verified complete/partial')
    if state['profile_hash'] != config['source_profile_hash']:
        raise ValueError('Source state profile mismatch')
    relative = state_path.relative_to(source_root)
    output = root/relative.parent.parent.parent/'data'/relative.parent.name/(state_path.stem+'.npz')
    target_state = root/relative
    # Construct from the authoritative state location; old archived absolute output paths are not trusted.
    source = source_root/relative.parent.parent.parent/'data'/relative.parent.name/(state_path.stem+'.npz')
    source_fp = fp(source)
    if sha(source) != state['output_sha256']:
        raise ValueError('Source moment NPZ SHA256 mismatch')
    inputs = [{'fingerprint': source_fp, 'sha256': state['output_sha256']},
              {'fingerprint': fp(state_path), 'sha256': sha(state_path)}]+support.inputs
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    common = {'profile_hash': identity, 'input_signature': signature, 'inputs': inputs,
              'start_utc': state['start_utc'], 'end_utc_exclusive': state['end_utc_exclusive']}
    if target_state.exists() and output.exists():
        old = json.loads(target_state.read_text())
        if (old.get('status') in ('complete', 'partial') and old.get('profile_hash') == identity
                and old.get('input_signature') == signature and old.get('output_sha256') == sha(output)):
            return 'skipped_verified' if old['status'] == 'complete' else 'skipped_partial'
    with np.load(source, allow_pickle=False) as source_data:
        times = source_data['time_unix'].copy()
        epoch = source_data['epoch_tt2000'].copy()
        original = json.loads(str(source_data['manifest_json']))
        if original['profile_hash'] != config['source_profile_hash'] or 'DSI' not in original['config']['frame']:
            raise ValueError('Source NPZ does not establish native DSI frame')
        if not len(times) or epoch.shape != times.shape or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError('Source moment timestamps invalid')
        if state['status'] == 'partial' and ('moment_valid' not in source_data or 'moment_invalid_reason' not in source_data):
            raise ValueError('Partial source lacks per-record moment validity')
        moment_valid = source_data['moment_valid'].copy() if 'moment_valid' in source_data else np.ones(len(times), dtype=bool)
        moment_reason = source_data['moment_invalid_reason'].copy() if 'moment_invalid_reason' in source_data else np.zeros(len(times), dtype=np.uint8)
        if (moment_valid.shape != times.shape or moment_valid.dtype.kind != 'b'
                or moment_reason.shape != times.shape or not np.array_equal(moment_valid, moment_reason == 0)
                or not moment_valid.any() or (state['status'] == 'complete' and not moment_valid.all())
                or (state['status'] == 'partial' and moment_valid.all())):
            raise ValueError('Source moment validity is inconsistent with state')
        matrices, diagnostics, brackets = rotation(times, support.at, support.angles,
            support.ot, support.positions, config['attitude_gap_seconds'], config['orbit_gap_seconds'])
        arrays = {'time_unix': times, 'epoch_tt2000': epoch, 'rotation_dsi_to_gsm': matrices,
                  'moment_valid': moment_valid, 'moment_invalid_reason': moment_reason, **diagnostics}
        for side in ('left', 'right'):
            q = support.quality[brackets[side]]
            arrays['attitude_'+side+'_flag_vi'] = q[:, 0].copy()
            arrays['attitude_'+side+'_sensor'] = q[:, 1].copy()
            arrays['attitude_'+side+'_flags_missing'] = (q[:, 0] == '') | (q[:, 1] == '')
            arrays['attitude_'+side+'_adsg_interp_dt_seconds'] = q[:, 2].astype(float)
            arrays['attitude_'+side+'_adsgt_iptime_seconds'] = q[:, 3].astype(float)
            unavailable = (diagnostics['rotation_quality'] & FLAGS['attitude_unavailable']) != 0
            arrays['attitude_'+side+'_flag_vi'][unavailable] = ''
            arrays['attitude_'+side+'_sensor'][unavailable] = ''
            arrays['attitude_'+side+'_adsg_interp_dt_seconds'][unavailable] = np.nan
            arrays['attitude_'+side+'_adsgt_iptime_seconds'][unavailable] = np.nan
        for var in SPECIES:
            velocity = source_data[var+'__velocity'].copy()
            velocity[~moment_valid] = np.nan
            observed, valid = apply(matrices, velocity)
            arrays[var+'__velocity_gsm_observed'] = observed
            arrays[var+'__velocity_valid'] = valid
    manifest = {**common, 'config': config, 'quality_bits': FLAGS,
                'source_moment_status': state['status'],
                'moment_selection': original.get('metadata', {}).get('moment_selection', {}),
                'rotation_convention': 'v_gsm[i] = R[i,j] v_dsi[j]; matrices stored per record',
                'attitude_quality': 'Original FLAG_VI, Sensor, ADSG_interp_dt, ADSGt-iptime retained at brackets; no flag/age filtering',
                'support_extrapolation': False, 'orbit_input_units': support.orbit_input_units, 'coordinate_reference': 'https://ergsc.isee.nagoya-u.ac.jp/assets/howto/ERG_Coordinate_System_202004.pdf',
                'array_sha256': {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key, value in arrays.items()}}
    check_code(config)
    save(output, arrays, manifest)
    rotation_incomplete = bool((moment_valid & ~diagnostics['rotation_valid']).any())
    status = 'complete' if moment_valid.all() and diagnostics['rotation_valid'].all() else 'partial'
    atomic_json(target_state, {**common, 'status': status, 'output': str(output),
                              'output_sha256': sha(output), 'records': len(times),
                              'valid_rotation_records': int(diagnostics['rotation_valid'].sum()),
                              'valid_source_moment_records': int(moment_valid.sum()),
                              'source_moment_status': state['status'], 'rotation_incomplete': rotation_incomplete})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moments-root', type=Path, required=True, help='Explicit native_dsi_<hash> profile root')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--month'); group.add_argument('--year', type=int); group.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--output-root', type=Path, default=Path('/mnt/j/observation_data/Arase_LEPi_GSM_velocity'))
    parser.add_argument('--max-attitude-gap', type=float, default=16.)
    parser.add_argument('--max-orbit-gap', type=float, default=120.)
    parser.add_argument('--earth-radius-km', type=float, default=6371.2, help='RE-to-km conversion, recorded in profile')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--download-attempts', type=int, default=3)
    parser.add_argument('--retry-delay', type=float, default=30.)
    parser.add_argument('--retry-passes', type=int, default=1)
    parser.add_argument('--pause-seconds', type=float, default=2.)
    parser.add_argument('--min-free-memory-gib', type=float, default=8.)
    parser.add_argument('--progress', choices=('auto','on','off'), default='auto')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.max_attitude_gap, args.max_orbit_gap,
            args.retry_delay,args.pause_seconds,args.min_free_memory_gib,args.earth_radius_km)) or min(args.max_attitude_gap,args.max_orbit_gap,args.earth_radius_km) <= 0 or args.download_attempts < 1 or args.retry_passes < 0:
        parser.error('Invalid resource/acquisition/interpolation options')
    try:
        start,end = selected_range(args)
        source_profile = json.loads((args.moments_root/'profile.json').read_text())
        identity, config = profile(args,source_profile)
        items = inventory(args.moments_root,start,end)
    except (OSError,ValueError,KeyError) as exc:
        parser.error(str(exc))
    root = args.output_root/('gsm_observed_'+identity[:12])
    if args.dry_run:
        print(json.dumps({'source_usable_chunks':len(items),'output_root':str(root),
                          'source_profile_hash':source_profile['profile_hash']},indent=2)); return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    os.nice(19); os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
    if shutil.which('ionice'):
        subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
    root.mkdir(parents=True,exist_ok=True)
    counts=Counter(); statuses={}; failures=[]
    report={'profile_hash':identity,'source_start_utc':start.isoformat(),'source_end_utc_exclusive':end.isoformat(),
            'inventory_scope':'Verified complete/partial source chunks present at startup; source batch may still be running'}
    try:
        with (root/'.run.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            atomic_json(root/'profile.json',{'profile_hash':identity,'config':config})
            atomic_json(root/'last_run.json',{**report,'status':'running'})
            from tqdm import tqdm
            support=Support(args)
            pending=items
            for number in range(1+(0 if args.offline else args.retry_passes)):
                if not pending: break
                failures=[]; failed_days={}
                if number: support.day=None
                enabled=args.progress=='on' or (args.progress=='auto' and sys.stderr.isatty())
                with tqdm(pending,desc=f'GSM velocity pass {number}',unit='chunk',disable=not enabled) as bar:
                    for lo,path in bar:
                        day=lo.replace(hour=0,minute=0,second=0,microsecond=0)
                        relative=path.relative_to(args.moments_root)
                        try:
                            check_code(config)
                            while available_gib() < args.min_free_memory_gib: time.sleep(10)
                            if day in failed_days: raise RuntimeError(failed_days[day])
                            try: support.ensure(day)
                            except Exception as exc:
                                failed_days[day]=str(exc); raise
                            status=convert_chunk(path,args.moments_root,root,identity,config,support)
                            if status in ('partial', 'skipped_partial'):
                                target = json.loads((root/relative).read_text())
                                if target.get('rotation_incomplete', True): failures.append((lo,path))
                        except Exception as exc:
                            status='failed'
                            atomic_json(root/relative,{'status':status,'profile_hash':identity,'reason':str(exc),
                                'source_state':str(path),'start_utc':lo.isoformat(),
                                'updated_utc':dt.datetime.now(UTC).isoformat()})
                            failures.append((lo,path))
                            print(f'{lo.isoformat()} failed: {exc}',flush=True)
                        previous=statuses.get(str(path))
                        if previous: counts[previous]-=1
                        statuses[str(path)]=status; counts[status]+=1
                        if enabled: bar.set_postfix(dict(counts),refresh=False)
                        time.sleep(args.pause_seconds)
                pending=failures
            counts=Counter({k:v for k,v in counts.items() if v})
            atomic_json(root/'last_run.json',{**report,'status':'finished','counts':dict(counts),
                        'updated_utc':dt.datetime.now(UTC).isoformat()})
    except BlockingIOError:
        print('Another GSM converter holds this profile lock',file=sys.stderr); return 3
    except KeyboardInterrupt:
        atomic_json(root/'last_run.json',{**report,'status':'interrupted','counts':dict(counts)})
        return 130
    print(json.dumps({'output_root':str(root),'counts':dict(counts)}))
    return 2 if failures else 0


if __name__=='__main__':
    sys.exit(main())
