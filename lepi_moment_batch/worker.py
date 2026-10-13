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

START = time.monotonic()
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'lepi_moment_pilot'))
# Import also installs the validated 1-CPU, 1-thread, nice=19, 3-GiB/120-s limits.
import pilot
np, CDF, psp = pilot.np, pilot.CDF, pilot.psp
from batch import atomic_json, sha_file, fingerprint


class MissingMagneticData(Exception):
    pass


def unique_order(epoch):
    """Sort TT2000, prefer last source at a duplicated granule timestamp."""
    _, reversed_indices = np.unique(epoch[::-1], return_index=True)
    keep = len(epoch) - 1 - reversed_indices
    return keep[np.argsort(epoch[keep])]


def times(cdf, name):
    """Validated native epochs may contain identical duplicates; selection keeps original indices."""
    raw = np.asarray(cdf.varget(name))
    dates = np.asarray(pilot.cdfepoch.to_datetime(raw)).astype('datetime64[ns]')
    if np.isnat(dates).any() or np.any(np.diff(raw) < 0):
        raise ValueError('Invalid or reversed epochs in verified input')
    return raw, dates, dates.astype('int64') / 1e9


def particle_records(staged, lo, hi):
    records = []
    for source_index, source in enumerate(staged):
        cdf = CDF(source['expanded_path'])
        epoch, dates, unix = times(cdf, 'Epoch')
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
        epoch, dates, unix = times(cdf, epoch_name)
        first = max(0, int(np.searchsorted(dates, lo)) - 1)
        last = min(len(dates) - 1, int(np.searchsorted(dates, hi)))
        if first > last:
            continue
        b, ba = pilot.record_data(cdf, field_name, first, last)
        b[(b < -1e6) | (b > 1e6)] = np.nan
        variables = cdf.cdf_info().zVariables
        if quality_name in variables:
            q, qa = pilot.record_data(cdf, quality_name, first, last, floating_fill=False)
        else:
            raise MissingMagneticData(f'{quality_name} unavailable in {source["source"]["path"]}')
        parts.append({'epoch': epoch[first:last+1], 'time': unix[first:last+1], 'b': b,
                      'q': q, 'source_index': np.full(last-first+1, source_index, dtype=np.int16)})
        attrs.append({'source_index': source_index, 'magnetic': ba, 'quality': qa})
    if not parts:
        raise MissingMagneticData('No local magnetic field samples')
    epoch = np.concatenate([p['epoch'] for p in parts])
    order = unique_order(epoch)
    merged = {key: np.concatenate([p[key] for p in parts])[order] for key in parts[0]}
    t, b = merged['time'], merged['b']
    if len(t) < 2 or targets[0] < t[0] or targets[-1] > t[-1]:
        raise MissingMagneticData('MGF does not bracket all particle timestamps')
    right = np.searchsorted(t, targets, side='left')
    right = np.minimum(right, len(t) - 1)
    left = np.maximum(right - 1, 0)
    exact = t[right] == targets
    left[exact] = right[exact]
    gaps = t[right] - t[left]
    good = (np.isfinite(b[left]).all(axis=1) & np.isfinite(b[right]).all(axis=1)
            & (gaps <= max_gap))
    if not good.all():
        raise MissingMagneticData(f'Invalid magnetic brackets: {int((~good).sum())} records; '
                                  f'max gap {float(gaps.max()):g} s')
    b_at = np.stack([np.interp(targets, t, b[:, k]) for k in range(3)], axis=1)
    if np.any(np.linalg.norm(b_at, axis=1) == 0):
        raise MissingMagneticData('Zero interpolated magnetic field')
    psp.store_data('batch_mgf_dsi', data={'x': t, 'y': b})
    arrays = {'mgf_epoch_tt2000': merged['epoch'], 'mgf_time_unix': t,
              'mgf_b_dsi': b, 'mgf_quality_raw': merged['q'],
              'mgf_source_file_index': merged['source_index'],
              'mgf_bracket_gap_seconds': gaps}
    return arrays, attrs


def species_data(records, order, var, targets):
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
        return {'status': 'no_data', 'input_verification': 'verified',
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
    metadata = {'magnetic_input': bmeta, 'species': {}, 'outputs': {},
                'tensor6_order': ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'],
                'extra_units': {'ttens': 'eV', 'vthermal': 'km/s', 'symm': 'dimensionless'},
                'quality': 'Raw flags retained; no flag selection applied',
                'duplicate_epochs': 'Identical duplicates within a granule: retain last original record; across granules: last dated source',
                'magt3_basis': 'Internal moments_3d magnetic basis; not a fixed geographic FAC'}
    for var, species in config['species'].items():
        name, diagnostics, smeta = species_data(records, order, var, targets)
        arrays.update(diagnostics)
        names = pilot.erg_lep_part_products(name, species=species, outputs=['moments'],
                                           energy=config['energy_range_ev_per_q'],
                                           mag_name='batch_mgf_dsi')
        if set(names) != {name + '_' + key for key in pilot.MOMENTS}:
            raise RuntimeError('Incomplete public moment outputs')
        metadata['species'][var] = smeta
        for key in pilot.MOMENTS:
            out = psp.get_data(name + '_' + key)
            np.testing.assert_array_equal(out.times, targets)
            arrays[var + '__' + key] = out.y.copy()
            metadata['outputs'][var + '__' + key] = psp.get_data(name + '_' + key, metadata=True)
        arrays[var + '__b_dsi_at_moment'] = psp.get_data('batch_mgf_dsi_pgs_temp').y.copy()
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
    return {'status': 'complete', 'records_per_species': len(targets), 'moment_series': 48,
            'output': str(output), 'file_bytes': output.stat().st_size,
            'output_sha256': sha_file(output),
            'finite_moment_fraction': {k: float(np.isfinite(v).mean()) for k, v in arrays.items()
                                       if k.split('__')[-1] in pilot.MOMENTS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    try:
        result = execute(request)
    except MissingMagneticData as exc:
        result = {'status': 'missing_mgf', 'reason': str(exc)}
    result.update(total_seconds=time.monotonic() - START,
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  nice=os.getpriority(os.PRIO_PROCESS, 0),
                  cpu_affinity=sorted(os.sched_getaffinity(0)))
    atomic_json(args.result, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
