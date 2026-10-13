"""Bounded, local-CDF-only LEP-i moment experiment, independent of KAW.

This is a pilot, not a mission-wide downloader or calibrated science product.
Each invocation is one short interval in a fresh process. No input file is written.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
import zlib
import datetime
import logging

START = time.monotonic()
# Apply limits before scientific-library imports. Existing jobs are unaffected.
os.nice(19)
os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lepi_moment_pilot_mpl')
os.environ.setdefault('SPACEPY', '/tmp/lepi_moment_pilot_spacepy')
Path(os.environ['SPACEPY']).mkdir(parents=True, exist_ok=True)

import numpy as np
from cdflib import CDF, cdfepoch
import pyspedas as psp
from pyspedas.projects.erg.satellite.erg.particle.erg_lep_part_products import erg_lep_part_products
logging.getLogger().setLevel(logging.WARNING)

IMPORT_SECONDS = time.monotonic() - START
SPECIES = {'FPDU': 'proton', 'FHEDU': 'heplus', 'FODU': 'oplus'}
MOMENTS = ['density', 'flux', 'mftens', 'velocity', 'ptens', 'ttens',
           'vthermal', 'avgtemp', 'eflux', 't3', 'magt3', 'symm',
           'symm_phi', 'symm_theta', 'symm_ang', 'qflux']
LEPI_DEFAULT = '/mnt/j/observation_data/ergsc/satellite/erg/lepi/l2/3dflux/2022/09/erg_lepi_l2_3dflux_20220929_v03_00.cdf'
MGF_DEFAULT = '/mnt/j/observation_data/ergsc/satellite/erg/mgf/l2/64hz/2022/09/erg_mgf_l2_64hz_dsi_2022092905_v04.06.cdf'


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sources():
    names = ['erg_lep_part_products', 'erg_lepi_get_dist', 'erg_pgs_clean_data',
             'erg_pgs_limit_range', 'erg_convert_flux_units',
             'get_lepi_flux_angle_in_sga', 'erg_pgs_moments_tplot']
    modules = ['pyspedas.projects.erg.satellite.erg.particle.' + n for n in names]
    modules += ['pyspedas.particles.moments.spd_pgs_moments',
                'pyspedas.particles.moments.moments_3d',
                'pyspedas.particles.moments.moments_3d_omega_weights']
    result = {}
    for name in modules:
        path = Path(importlib.import_module(name).__file__)
        result[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    path = Path(__file__)
    result['pilot'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def fingerprint(path):
    path = Path(path).resolve()
    st = path.stat()
    return {'path': str(path), 'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def prepare_cdf(path):
    """Stream file-level CDF3 gzip to a reusable local scratch file.

    cdflib's default gzip.decompress allocates the entire expanded day at once.
    This routine validates gzip CRC/end, CCR expanded length, and stores SHA256.
    It never changes cdflib or the original CDF. Other compression is rejected.
    """
    source = Path(path).resolve()
    with source.open('rb') as f:
        head = f.read(40)
        if head[4:8] == bytes.fromhex('0000ffff'):
            return source, {'file_level_compression': False}
        if head[:4] != bytes.fromhex('cdf30001'):
            raise ValueError('Streaming pilot supports CDF3 only')
        block_size = int.from_bytes(head[8:16], 'big')
        cpr_offset = int.from_bytes(head[20:28], 'big')
        expanded_size = int.from_bytes(head[28:36], 'big')
        f.seek(cpr_offset + 12)
        if int.from_bytes(f.read(4), 'big') != 5:
            raise ValueError('Streaming pilot supports file-level gzip only')
    source_fp = fingerprint(source)
    key = hashlib.sha256(json.dumps(source_fp, sort_keys=True).encode()).hexdigest()
    cache = Path('/tmp/lepi_moment_pilot_cdf')
    cache.mkdir(exist_ok=True)
    target = cache / (key + '.cdf')
    marker = target.with_suffix('.json')
    if target.exists() and marker.exists():
        meta = json.loads(marker.read_text())
        if meta['source'] == source_fp and target.stat().st_size == expanded_size + 8:
            return target, meta
        raise RuntimeError('Invalid expanded-CDF cache; inspect before reuse')
    temp = target.with_suffix('.cdf.tmp')
    sha = hashlib.sha256()
    written = 0
    try:
        with source.open('rb') as f, temp.open('xb') as out:
            header = bytes.fromhex('cdf300010000ffff')
            out.write(header)
            sha.update(header)
            f.seek(40)
            remaining = block_size - 32
            dec = zlib.decompressobj(wbits=31)
            while remaining:
                block = f.read(min(1024**2, remaining))
                if not block:
                    raise EOFError('Truncated compressed CDF')
                remaining -= len(block)
                while block:
                    expanded = dec.decompress(block, 1024**2)
                    out.write(expanded)
                    sha.update(expanded)
                    written += len(expanded)
                    block = dec.unconsumed_tail
                if dec.unused_data:
                    raise ValueError('Unexpected trailing data in CDF gzip payload')
            if not dec.eof or written != expanded_size:
                raise ValueError('Incomplete gzip or CCR expanded-size mismatch')
            out.flush()
            os.fsync(out.fileno())
        temp.rename(target)
        meta = {'source': source_fp, 'file_level_compression': 'gzip',
                'expanded_path': str(target), 'expanded_bytes': written + 8,
                'expanded_sha256': sha.hexdigest(), 'gzip_crc_and_length_verified': True}
        marker.write_text(json.dumps(meta, indent=2))
        return target, meta
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def times(cdf, epoch):
    raw = cdf.varget(epoch)
    dt = np.asarray(cdfepoch.to_datetime(raw)).astype('datetime64[ns]')
    if np.any(np.diff(dt.astype('int64')) <= 0) or np.any(np.isnat(dt)):
        raise ValueError('Nonfinite/non-increasing epoch; explicit handling required')
    return raw, dt, dt.astype('int64') / 1e9


def record_data(cdf, var, first, last, floating_fill=True):
    data = np.array(cdf.varget(var, startrec=first, endrec=last), copy=True)
    attrs = cdf.varattsget(var)
    # Match cdf_to_tplot: floating FILLVAL -> NaN. Preserve quality bytes verbatim.
    if floating_fill and np.issubdtype(data.dtype, np.floating) and 'FILLVAL' in attrs:
        data[data == float(attrs['FILLVAL'])] = np.nan
    return data, attrs


def load_magnetic(path, lo, hi):
    cdf = CDF(path)
    raw, dt, unix = times(cdf, 'epoch_64hz')
    first = max(0, int(np.searchsorted(dt, lo)) - 1)
    last = min(len(dt) - 1, int(np.searchsorted(dt, hi)))
    if unix[first] > lo.astype('int64') / 1e9 or unix[last] < hi.astype('int64') / 1e9:
        raise ValueError('MGF CDF does not bracket interval')
    b, attrs = record_data(cdf, 'mag_64hz_dsi', first, last)
    # Standard MGF loader's physical-range clipping.
    b[(b < -1e6) | (b > 1e6)] = np.nan
    if not np.isfinite(b).all() or np.any(np.linalg.norm(b, axis=1) == 0):
        raise ValueError('Pilot requires finite, nonzero bracketing magnetic field')
    q, qa = record_data(cdf, 'quality_64hz', first, last, floating_fill=False)
    # Keep raw quality; interpretation/masking is a PI discussion, not inferred here.
    if not psp.store_data('pilot_mgf_dsi', data={'x': unix[first:last+1], 'y': b}):
        raise RuntimeError('Failed to store magnetic field')
    return {'mgf_epoch_tt2000': raw[first:last+1], 'mgf_time_unix': unix[first:last+1],
            'mgf_b_dsi': b, 'mgf_quality_raw': q}, {'magnetic': attrs, 'quality': qa}


def load_species(cdf, var, first, last, unix, standard_check):
    raw, attrs = record_data(cdf, var, first, last)
    if raw.shape[1:] != (32, 8, 16):
        raise ValueError(f'Unsupported native geometry: {raw.shape}')
    ename = attrs['DEPEND_1']
    if cdf.varinq(ename).Rec_Vary:
        raise ValueError('Time-dependent energy table needs explicit support')
    energy = np.asarray(cdf.varget(ename))[:30]
    name = 'erg_lepi_l2_3dflux_' + var
    data = {'x': unix, 'y': raw[:, :30], 'v1': energy,
            'v2': np.arange(8), 'v3': np.arange(16)}
    if standard_check:
        # Independent standard importer, then the exact public lepi() reshape.
        # Only this check reads one species for the full day; no full-day moments.
        imported = psp.cdf_to_tplot(str(cdf.file), varnames=[var],
                                    prefix='erg_lepi_l2_3dflux_')
        if name not in imported:
            raise RuntimeError('Standard CDF import failed')
        ref = psp.get_data(name)
        selection = slice(first, last + 1)
        for label, a, b in [('epoch', unix, ref.times[selection]),
                            ('flux', data['y'], ref.y[selection, :30]),
                            ('energy', energy, ref.v1[:30])]:
            np.testing.assert_array_equal(a, b, err_msg=f'{var}: {label} loader mismatch')
        del ref
    if not psp.store_data(name, data=data):
        raise RuntimeError('Failed to store particle distribution')
    counts, ca = record_data(cdf, var + '_COUNT_RAW', first, last)
    quality, qa = record_data(cdf, var + '_Quality', first, last, floating_fill=False)
    selected = (energy * 1000 >= 30) & (energy * 1000 <= 30000)
    selected[0] = False
    valid = np.isfinite(raw[:, :30, :, :])[:, selected]
    diagnostics = {
        var + '__energy_kev_per_q': energy,
        var + '__energy_selected': selected,
        var + '__counts_raw': counts[:, :30],
        var + '__quality_raw': quality[:, :30],
        var + '__finite_flux_fraction_selected': valid.mean(axis=(1, 2, 3)),
        var + '__negative_flux_bins_selected': (raw[:, :30][:, selected] < 0).sum(axis=(1, 2, 3)),
    }
    meta = {'flux': attrs, 'counts': ca, 'quality': qa, 'selected_flux_sha256': digest(data['y'])}
    return name, diagnostics, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2022-09-29T05:05:00')
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--lepi-cdf', default=LEPI_DEFAULT)
    parser.add_argument('--mgf-cdf', default=MGF_DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--standard-check', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900:
        parser.error('Pilot interval must be 1–900 seconds')
    lo = np.datetime64(args.start, 'ns')
    hi = lo + np.timedelta64(args.seconds, 's')
    config = {'start_utc': str(lo), 'end_utc_exclusive': str(hi),
              'energy_range_ev_per_q': [30., 30000.], 'outputs': MOMENTS,
              'species': SPECIES, 'inputs': [fingerprint(args.lepi_cdf), fingerprint(args.mgf_cdf)],
              'sources': sources(), 'standard_check': args.standard_check,
              'loader': 'direct CDF records; first 30 energy channels; v2=0..7; v3=0..15',
              'frame': 'DSI; spacecraft-relative; no spacecraft-velocity correction',
              'spacecraft_potential_volts_used': 0., 'quality_mask_applied': False,
              'sub_anodes_used': False, 'schema': 'lepi_moment_pilot_v1'}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as saved:
            previous = json.loads(str(saved['manifest_json']))
            if previous['config_hash'] != config_hash:
                raise RuntimeError('Existing output has a different configuration; use another path')
            for key in previous['array_sha256']:
                if digest(saved[key]) != previous['array_sha256'][key]:
                    raise RuntimeError('Existing output array checksum mismatch')
        print(json.dumps({'status': 'skipped_verified', 'output': str(args.output)}))
        return
    t0 = time.monotonic()
    lepi_path, lepi_cache = prepare_cdf(args.lepi_cdf)
    mgf_path, mgf_cache = prepare_cdf(args.mgf_cdf)
    prepare_seconds = time.monotonic() - t0
    cdf = CDF(str(lepi_path))
    raw_epoch, dt, unix = times(cdf, 'Epoch')
    first, stop = np.searchsorted(dt, [lo, hi])
    if first == stop:
        raise ValueError('No particle records in half-open interval')
    first, last = int(first), int(stop - 1)
    array, bmeta = load_magnetic(str(mgf_path), lo, hi)
    array.update(epoch_tt2000=raw_epoch[first:last+1], time_unix=unix[first:last+1])
    # Preserve sector times when available, without treating them as a spin cadence.
    array['epoch_sp_tt2000'] = cdf.varget('epoch_sp', startrec=first, endrec=last)
    metadata = {'magnetic_input': bmeta, 'species': {}, 'outputs': {},
                'expanded_cdf': {'lepi': lepi_cache, 'mgf': mgf_cache}}
    timings = {'import_seconds': IMPORT_SECONDS, 'prepare_cdf_seconds': prepare_seconds}
    for var, species in SPECIES.items():
        t0 = time.monotonic()
        name, diagnostics, smeta = load_species(cdf, var, first, last,
                                               unix[first:last+1], args.standard_check)
        array.update(diagnostics)
        timings[var + '_load_seconds'] = time.monotonic() - t0
        t0 = time.monotonic()
        names = erg_lep_part_products(name, species=species, outputs=['moments'],
                                      energy=[30., 30000.], mag_name='pilot_mgf_dsi')
        if set(names) != {name + '_' + key for key in MOMENTS}:
            raise RuntimeError(f'Incomplete moments: {names}')
        timings[var + '_compute_seconds'] = time.monotonic() - t0
        metadata['species'][var] = smeta
        for key in MOMENTS:
            out = psp.get_data(name + '_' + key)
            np.testing.assert_array_equal(out.times, array['time_unix'])
            array[var + '__' + key] = np.array(out.y, copy=True)
            metadata['outputs'][var + '__' + key] = psp.get_data(name + '_' + key, metadata=True)
        # Also store the field actually supplied to the moment integrator.
        array[var + '__b_dsi_at_moment'] = psp.get_data('pilot_mgf_dsi_pgs_temp').y.copy()
        psp.del_data('erg_lepi_l2_3dflux_' + var + '*')
    metadata['definitions'] = {
        'tensor6_order': ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
        'ttens_units': 'eV', 'vthermal_units': 'km/s', 'symm_units': 'dimensionless',
        'magt3_basis': 'internal magnetic basis of moments_3d; not a fixed geographic FAC',
        'quality_raw': 'Uninterpreted CDF byte flags; no quality selection applied',
        'negative_flux': 'Public getter replaces finite negative flux with zero',
        'window': 'Timestamp selection [start,end); complete spin records retained',
    }
    manifest = {'config': config, 'config_hash': config_hash, 'metadata': metadata,
                'packages': {k: importlib.metadata.version(k) for k in ['numpy', 'pyspedas', 'cdflib']},
                'array_sha256': {key: digest(a) for key, a in array.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.npz.tmp')
    t0 = time.monotonic()
    manifest_json = json.dumps(manifest, default=jsonable)
    with tmp.open('xb') as f:
        np.savez_compressed(f, **array, manifest_json=manifest_json)
        f.flush()
        os.fsync(f.fileno())
    # Verify every saved array before committing the artifact.
    with np.load(tmp, allow_pickle=False) as saved:
        for key, a in array.items():
            np.testing.assert_array_equal(saved[key], a, err_msg=key)
    tmp.rename(args.output)
    timings['save_verify_seconds'] = time.monotonic() - t0
    report = {'status': 'complete', 'output': str(args.output),
              'records_per_species': last - first + 1, 'moment_series': 3 * len(MOMENTS),
              'file_bytes': args.output.stat().st_size,
              'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              'total_seconds': time.monotonic() - START,
              'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
              'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'timings': timings, 'standard_loader_check': args.standard_check,
              'finite_moment_fraction': {k: float(np.isfinite(v).mean()) for k, v in array.items()
                                         if k.split('__')[-1] in MOMENTS}}
    args.output.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
