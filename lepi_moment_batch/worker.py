"""One five-minute worker; reuses the independently validated public pipeline."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import zipfile

START = time.monotonic()
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'lepi_moment_pilot'))
# Keep the same resource bounds before even the lightweight imports.
os.nice(19)
os.sched_setaffinity(0, {int(os.environ['LEPI_COMPUTE_CPU'])}
                     if 'LEPI_COMPUTE_CPU' in os.environ else {max(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from cdflib import CDF, cdfepoch
from batch import atomic_json, sha_file, fingerprint, MOMENT_INVALID_BITS, MOMENT_TIME_POLICY

pilot = psp = None


def load_science():
    global pilot, psp
    if pilot is None:
        import pilot as module
        pilot, psp = module, module.psp


def record_data(cdf, var, first, last, floating_fill=True):
    # Same fill-value rule as pilot.record_data; raw quality bytes stay unchanged.
    data = np.array(cdf.varget(var, startrec=first, endrec=last), copy=True)
    attrs = cdf.varattsget(var)
    if floating_fill and np.issubdtype(data.dtype, np.floating) and 'FILLVAL' in attrs:
        data[data == float(attrs['FILLVAL'])] = np.nan
    return data, attrs


class MissingMagneticData(Exception):
    def __init__(self, reason, failure_class='insufficient_mgf_coverage', **details):
        super().__init__(reason)
        self.failure_class, self.details = failure_class, details


def unique_order(epoch):
    """Sort TT2000, prefer last source at a duplicated granule timestamp."""
    _, reversed_indices = np.unique(epoch[::-1], return_index=True)
    keep = len(epoch) - 1 - reversed_indices
    return keep[np.argsort(epoch[keep])]


def times(cdf, name, source=None):
    """Validated native epochs may contain identical duplicates; selection keeps original indices."""
    cache = Path(source['time_cache_path']) if source and source.get('time_cache_path') else None
    stamp = fingerprint(source['expanded_path']) if cache else None
    if cache and cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                if json.loads(str(saved['fingerprint'])) == stamp and str(saved['epoch_name']) == name:
                    raw, dates = saved['epoch'], saved['dates']
                    return raw, dates, dates.astype('int64') / 1e9
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
    raw = np.asarray(cdf.varget(name))
    dates = np.asarray(cdfepoch.to_datetime(raw)).astype('datetime64[ns]')
    if np.isnat(dates).any() or np.any(np.diff(raw) < 0):
        raise ValueError('Invalid or reversed epochs in verified input')
    if cache:
        if fingerprint(source['expanded_path']) != stamp:
            raise RuntimeError('Expanded input changed during epoch indexing')
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=cache.parent, suffix='.npz')
        try:
            with os.fdopen(fd, 'wb') as out:
                np.savez(out, epoch=raw, dates=dates, epoch_name=name,
                         fingerprint=json.dumps(stamp, sort_keys=True))
            os.replace(temp, cache)
        finally:
            Path(temp).unlink(missing_ok=True)
    return raw, dates, dates.astype('int64') / 1e9


def particle_records(staged, lo, hi):
    records = []
    for source_index, source in enumerate(staged):
        cdf = CDF(source['expanded_path'])
        epoch, dates, unix = times(cdf, 'Epoch', source)
        first, stop = np.searchsorted(dates, [lo, hi])
        if first == stop:
            continue
        first, last = int(first), int(stop - 1)
        records.append({'cdf': cdf, 'source_index': source_index,
                        'first': first, 'last': last, 'unix': unix[first:last+1],
                        'epoch': epoch[first:last+1],
                        'sector_epoch': cdf.varget('epoch_sp', startrec=first, endrec=last)})
    return records


def magnetic(staged, lo, hi, targets, cadence, max_gap):
    epoch_name = 'epoch_64hz' if cadence == '64hz' else 'epoch_8sec'
    field_name = 'mag_64hz_dsi' if cadence == '64hz' else 'mag_8sec_dsi'
    quality_name = 'quality_64hz' if cadence == '64hz' else 'quality_8sec'
    parts, attrs = [], []
    for source_index, source in enumerate(staged):
        cdf = CDF(source['expanded_path'])
        epoch, dates, unix = times(cdf, epoch_name, source)
        first = max(0, int(np.searchsorted(dates, lo)) - 1)
        last = min(len(dates) - 1, int(np.searchsorted(dates, hi)))
        if first > last:
            continue
        b, ba = record_data(cdf, field_name, first, last)
        b[(b < -1e6) | (b > 1e6)] = np.nan
        variables = cdf.cdf_info().zVariables
        if quality_name in variables:
            q, qa = record_data(cdf, quality_name, first, last, floating_fill=False)
        else:
            raise MissingMagneticData(f'{quality_name} unavailable in {source["source"]["path"]}', 'invalid_mgf_schema')
        parts.append({'epoch': epoch[first:last+1], 'time': unix[first:last+1], 'b': b,
                      'q': q, 'source_index': np.full(last-first+1, source_index, dtype=np.int16)})
        attrs.append({'source_index': source_index, 'magnetic': ba, 'quality': qa})
    if not parts:
        raise MissingMagneticData('No local magnetic field samples', 'empty_mgf_data')
    epoch = np.concatenate([p['epoch'] for p in parts])
    order = unique_order(epoch)
    merged = {key: np.concatenate([p[key] for p in parts])[order] for key in parts[0]}
    t, b = merged['time'], merged['b']
    inside = (targets >= t[0]) & (targets <= t[-1]) & (len(t) >= 2)
    right = np.minimum(np.searchsorted(t, targets, side='left'), len(t) - 1)
    left = np.maximum(right - 1, 0)
    exact = t[right] == targets
    left[exact] = right[exact]
    gaps = t[right] - t[left]
    finite = np.isfinite(b[left]).all(axis=1) & np.isfinite(b[right]).all(axis=1)
    reason = np.zeros(len(targets), dtype=np.uint8)
    reason[~inside] |= MOMENT_INVALID_BITS['outside_mgf_support']
    reason[inside & ~finite] |= MOMENT_INVALID_BITS['nonfinite_mgf_bracket']
    reason[inside & (gaps > max_gap)] |= MOMENT_INVALID_BITS['mgf_gap_exceeded']
    b_at = np.full((len(targets), 3), np.nan)
    good = reason == 0
    if good.any():
        b_at[good] = np.stack([np.interp(targets[good], t, b[:, k]) for k in range(3)], axis=1)
    zero = good & (np.linalg.norm(b_at, axis=1) == 0)
    reason[zero] |= MOMENT_INVALID_BITS['zero_interpolated_mgf']
    b_at[reason != 0] = np.nan
    gaps = gaps.astype(float)
    gaps[~inside] = np.nan
    good = reason == 0
    if not good.any():
        invalid_field = bool((reason & (MOMENT_INVALID_BITS['nonfinite_mgf_bracket']
                                       | MOMENT_INVALID_BITS['zero_interpolated_mgf'])).any())
        raise MissingMagneticData('No particle timestamps have valid magnetic support',
                                  'invalid_mgf_field' if invalid_field else 'insufficient_mgf_coverage',
                                  invalid_brackets=len(targets),
                                  max_gap_seconds=float(np.max(gaps[inside])) if inside.any() else None,
                                  moment_invalid_reason_counts=reason_counts(reason))
    load_science()
    # Eligibility is fixed using the original brackets above. Removing invalid
    # samples here cannot make a rejected particle timestamp eligible again.
    # It prevents xarray interpolation propagating a neighbouring NaN even at
    # an exact finite sample. Preserve the original magnetic samples in arrays.
    keep = np.isfinite(b).all(axis=1)
    public_t, public_b = t[keep], b[keep]
    if len(public_t) == 1:
        # Only exact matches at this one measured timestamp can be eligible.
        # Two constant guard points satisfy the public linear interpolator's
        # two-point requirement; no other particle timestamp is sent to it.
        public_t = t[[0, -1]]
        public_b = np.repeat(public_b, 2, axis=0)
    psp.store_data('batch_mgf_dsi', data={'x': public_t, 'y': public_b})
    arrays = {'mgf_epoch_tt2000': merged['epoch'], 'mgf_time_unix': t,
              'mgf_b_dsi': b, 'mgf_quality_raw': merged['q'],
              'mgf_source_file_index': merged['source_index'],
              'mgf_bracket_gap_seconds': gaps, 'moment_valid': good,
              'moment_invalid_reason': reason}
    return arrays, attrs


def reason_counts(reason):
    return {key: int(((reason & bit) != 0).sum()) for key, bit in MOMENT_INVALID_BITS.items()}


def species_data(records, order, var, targets):
    load_science()
    flux, diagnostics, metadata = [], [], []
    energy = None
    for r in records:
        name, diag, meta = pilot.load_species(r['cdf'], var, r['first'], r['last'],
                                              r['unix'], standard_check=False)
        this_energy = diag[var + '__energy_kev_per_q']
        if energy is not None:
            np.testing.assert_array_equal(energy, this_energy,
                                          err_msg='Energy table changes inside chunk')
        energy = this_energy
        flux.append(psp.get_data(name).y.copy())
        diagnostics.append(diag)
        metadata.append({'source_index': r['source_index'], **meta})
    arrays = {}
    for key in diagnostics[0]:
        if key.endswith('__energy_kev_per_q') or key.endswith('__energy_selected'):
            arrays[key] = diagnostics[0][key]
        else:
            arrays[key] = np.concatenate([d[key] for d in diagnostics])[order]
    name = 'erg_lepi_l2_3dflux_' + var
    psp.store_data(name, data={'x': targets, 'y': np.concatenate(flux)[order],
                              'v1': energy, 'v2': np.arange(8), 'v3': np.arange(16)})
    return name, arrays, metadata


def check_input_files(request):
    for source in request['lepi'] + request['mgf']:
        if fingerprint(source['source']['path']) != source['source']:
            raise RuntimeError('Verified source changed during calculation: ' + source['source']['path'])


def execute(request):
    if request.get('input_verification') != 'verified':
        raise RuntimeError('Refusing unverified acquisition inputs')
    check_input_files(request)
    config = request['config']
    lo = np.datetime64(request['start_utc'].replace('+00:00', ''), 'ns')
    hi = np.datetime64(request['end_utc_exclusive'].replace('+00:00', ''), 'ns')
    records = particle_records(request['lepi'], lo, hi)
    if not records:
        check_input_files(request)
        return {'status': 'excluded', 'failure_class': 'empty_lepi_data', 'retryable': False,
                'classification_version': 1, 'exclusion_basis': 'verified_cdf_timestamps',
                'input_verification': 'verified',
                'reason': 'Verified LEP-i CDF has no spin timestamps in interval'}
    epoch = np.concatenate([r['epoch'] for r in records])
    order = unique_order(epoch)
    arrays = {
        'epoch_tt2000': epoch[order],
        'time_unix': np.concatenate([r['unix'] for r in records])[order],
        'epoch_sp_tt2000': np.concatenate([r['sector_epoch'] for r in records])[order],
        'lepi_source_file_index': np.concatenate([np.full(len(r['epoch']), r['source_index'], dtype=np.int16)
                                                for r in records])[order],
        'lepi_source_record_index': np.concatenate([np.arange(r['first'], r['last']+1)
                                                  for r in records])[order],
    }
    targets = arrays['time_unix']
    b, bmeta = magnetic(request['mgf'], lo, hi, targets, config['mgf_cadence'],
                        config['max_mgf_gap_seconds'])
    arrays.update(b)
    valid = arrays['moment_valid']
    computed = int(valid.sum())
    selection = {'policy': MOMENT_TIME_POLICY, 'invalid_reason_bits': MOMENT_INVALID_BITS,
                 'records': len(targets), 'computed_records': computed,
                 'invalid_records': int((~valid).sum()), 'invalid_reason_counts': reason_counts(arrays['moment_invalid_reason']),
                 'note': 'Validity describes magnetic support, not validity of all particle-distribution bins',
                 'public_magnetic_input': 'Finite samples only after original-bracket selection; singleton exact matches use constant interpolation guards'}
    metadata = {'magnetic_input': bmeta, 'moment_selection': selection, 'species': {}, 'outputs': {},
                'tensor6_order': ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
                'extra_units': {'ttens': 'eV', 'vthermal': 'km/s', 'symm': 'dimensionless'},
                'quality': 'Raw flags retained; no flag selection applied',
                'duplicate_epochs': 'Identical duplicates within a granule: retain last original record; across granules: last dated source',
                'magt3_basis': 'Internal moments_3d magnetic basis; not a fixed geographic FAC'}
    for var, species in config['species'].items():
        name, diagnostics, smeta = species_data(records, order, var, targets)
        arrays.update(diagnostics)
        if not valid.all():
            distribution = psp.get_data(name)
            psp.store_data(name, data={'x': targets[valid], 'y': distribution.y[valid],
                                      'v1': diagnostics[var + '__energy_kev_per_q'],
                                      'v2': np.arange(8), 'v3': np.arange(16)})
        names = pilot.erg_lep_part_products(name, species=species, outputs=['moments'],
                                           energy=config['energy_range_ev_per_q'],
                                           mag_name='batch_mgf_dsi')
        if set(names) != {name + '_' + key for key in pilot.MOMENTS}:
            raise RuntimeError('Incomplete public moment outputs')
        metadata['species'][var] = smeta
        for key in pilot.MOMENTS:
            out = psp.get_data(name + '_' + key)
            np.testing.assert_array_equal(out.times, targets[valid])
            if valid.all():
                arrays[var + '__' + key] = out.y.copy()
            else:
                padded = np.full((len(targets), *out.y.shape[1:]), np.nan, dtype=out.y.dtype)
                padded[valid] = out.y
                arrays[var + '__' + key] = padded
            metadata['outputs'][var + '__' + key] = psp.get_data(name + '_' + key, metadata=True)
        moment_b = np.full((len(targets), 3), np.nan)
        moment_b[valid] = psp.get_data('batch_mgf_dsi_pgs_temp').y
        arrays[var + '__b_dsi_at_moment'] = moment_b
        psp.del_data('erg_lepi_l2_3dflux_' + var + '*')
    manifest = {'profile_hash': request['profile_hash'], 'input_signature': request['input_signature'],
                'config': config, 'start_utc': request['start_utc'],
                'end_utc_exclusive': request['end_utc_exclusive'], 'metadata': metadata,
                'input_cdf': {'lepi': request['lepi'], 'mgf': request['mgf']},
                'acquisition_evidence': request['acquisition_evidence'],
                'input_verification': request['input_verification'],
                'array_sha256': {key: pilot.digest(a) for key, a in arrays.items()}}
    manifest_json = json.dumps(manifest, default=pilot.jsonable)
    output = Path(request['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays, manifest_json=manifest_json)
            f.flush()
            os.fsync(f.fileno())
        with np.load(tmp, allow_pickle=False) as saved:
            for key, a in arrays.items():
                np.testing.assert_array_equal(saved[key], a, err_msg=key)
        check_input_files(request)
        os.replace(tmp, output)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {'status': 'complete' if valid.all() else 'partial',
            'records_per_species': len(targets), 'computed_records_per_species': computed,
            'invalid_records_per_species': int((~valid).sum()), 'valid_moment_fraction': float(valid.mean()),
            'moment_invalid_reason_counts': selection['invalid_reason_counts'], 'retryable': False,
            'moment_series': 48,
            'output': str(output), 'file_bytes': output.stat().st_size,
            'output_sha256': sha_file(output),
            'finite_moment_fraction': {k: float(np.isfinite(v).mean()) for k, v in arrays.items()
                                       if k.split('__')[-1] in pilot.MOMENTS}}


def magnetic_failure_result(request, exc):
    invalid = exc.failure_class in {'invalid_mgf_schema', 'invalid_mgf_field'}
    remote = (not invalid and exc.failure_class == 'empty_mgf_data'
              and not request['mgf'] and any(x['unit'].startswith(('64hz_', '8sec_'))
                  and x['status'] == 'remote_absent' for x in request['acquisition_evidence']))
    if exc.failure_class == 'empty_mgf_data' and not request['mgf'] and not remote:
        return {'status': 'failed_validation', 'failure_class': 'unverified_mgf_absence',
                'reason': str(exc), 'retryable': True}
    result = {'status': 'failed_validation' if invalid else 'excluded',
              'failure_class': 'required_input_missing_remote' if remote else exc.failure_class,
              'reason': str(exc), 'retryable': False, 'classification_version': 1,
              **exc.details}
    if not invalid:
        result['exclusion_basis'] = 'verified_remote_index' if remote else 'verified_mgf_coverage'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    try:
        result = execute(request)
    except MissingMagneticData as exc:
        result = magnetic_failure_result(request, exc)
        check_input_files(request)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {'status': 'failed', 'failure_class': 'calculation_error', 'retryable': False,
                  'reason': str(exc), 'exception_type': type(exc).__name__}
    result.update(total_seconds=time.monotonic() - START,
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  nice=os.getpriority(os.PRIO_PROCESS, 0),
                  cpu_affinity=sorted(os.sched_getaffinity(0)))
    atomic_json(args.result, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
