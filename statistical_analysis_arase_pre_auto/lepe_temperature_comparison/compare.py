"""Independent LEP-e partial-moment / PI bi-Maxwellian temperature comparison."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import zlib

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lepe_temperature_mpl')
os.environ.setdefault('SPACEPY', '/tmp/lepe_temperature_spacepy')

import numpy as np
import pandas as pd

ANALYSIS_ROOT = Path('/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation/lepe_ge67eV_vion_v2')
RAW_ROOT = Path('/mnt/j/observation_data/ergsc/satellite/erg')
FIT_ROOT = Path('/mnt/j/observation_data/lepe-fitmom-data-v2/1min')
PREFIX = 'erg_lepe_l2_3dflux_FEDU'
STEM = 'lepe_temperature_comparison_dual_energy'
OUTPUT_DIR = 'lepe_temperature_comparison'
COMPONENTS = ('effective', 'parallel', 'perpendicular')
PROFILES = {
    'all': dict(upper=np.inf, label='Integral: E >= 67 eV', color='C0'),
    'le1000': dict(upper=1000., label='Integral: 67 <= E <= 1000 eV', color='C2'),
}
OUTPUT_SUFFIXES = ('.png', '.pdf', '_native.csv', '_1min.csv', '_diagnostics.npz')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


@contextmanager
def staged_cdf(source, scratch):
    """Stream CDF3 gzip, with CRC/length checks; never alter the raw CDF."""
    source = Path(source)
    with source.open('rb') as f:
        header = f.read(40)
    if header[4:8] == bytes.fromhex('0000ffff'):
        yield source
        return
    if header[:8] != bytes.fromhex('cdf30001cccc0001'):
        raise ValueError(f'Unsupported CDF header: {source}')
    size = int.from_bytes(header[8:16], 'big') - 32
    expected = int.from_bytes(header[28:36], 'big')
    cpr = int.from_bytes(header[20:28], 'big')
    with tempfile.TemporaryDirectory(prefix='lepe_temperature_', dir=scratch) as tmp:
        target = Path(tmp) / source.name
        with source.open('rb') as inp, target.open('wb') as out:
            inp.seek(cpr + 12)
            if int.from_bytes(inp.read(4), 'big') != 5:
                raise ValueError('Only gzip-compressed CDF3 is supported')
            inp.seek(40)
            out.write(bytes.fromhex('cdf300010000ffff'))
            dec, count = zlib.decompressobj(31), 0
            while size:
                block = inp.read(min(size, 1024**2))
                if not block:
                    raise EOFError('Truncated CDF payload')
                size -= len(block)
                while block:
                    expanded = dec.decompress(block, 1024**2)
                    out.write(expanded)
                    count += len(expanded)
                    block = dec.unconsumed_tail
                if dec.unused_data or count > expected:
                    raise ValueError('Invalid CDF gzip payload')
            if not dec.eof or count != expected:
                raise ValueError('Incomplete CDF gzip payload')
        yield target


def project_temperature(tensor, b):
    """DSI tensor -> field-parallel and mean of the two perpendicular axes."""
    tensor, b = np.asarray(tensor), np.asarray(b)
    norm = np.linalg.norm(b, axis=1)
    bhat = np.divide(b, norm[:, None], out=np.full_like(b, np.nan, dtype=float),
                     where=norm[:, None] > 0)
    trace = np.trace(tensor, axis1=1, axis2=2)
    parallel = np.einsum('ni,nij,nj->n', bhat, tensor, bhat)
    return np.column_stack((trace / 3, parallel, (trace - parallel) / 2))


def field_at_times(time, b, targets, max_gap=16.):
    """No extrapolation or interpolation across magnetic gaps > max_gap seconds."""
    time, targets = np.asarray(time), np.asarray(targets)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('Magnetic timestamps must be strictly increasing')
    right = np.searchsorted(time, targets, side='left').clip(1, len(time) - 1)
    left = right - 1
    gap = time[right] - time[left]
    exact = (targets == time[left]) | (targets == time[right])
    valid = ((targets >= time[0]) & (targets <= time[-1]) & ((gap <= max_gap) | exact)
             & np.isfinite(b[left]).all(axis=1) & np.isfinite(b[right]).all(axis=1))
    out = np.column_stack([np.interp(targets, time, b[:, i]) for i in range(3)])
    valid &= np.linalg.norm(out, axis=1) > 0
    out[~valid] = np.nan
    return out, valid, gap, left, right


@lru_cache(maxsize=4)
def range_states(states):
    """Snapshot each dataset once per process, avoiding repeated monthly JSON reads."""
    return tuple((marker, json.loads(marker.read_text())) for marker in sorted(states.glob('range_*.json')))


def discover_events(dataset, date):
    """Use exact batch-state timestamps, never infer seconds from folder labels."""
    states = dataset.parent.parent / 'run_state' / dataset.name / 'ranges'
    events = []
    for marker, state in range_states(states):
        start, end = pd.Timestamp(state['start_time']), pd.Timestamp(state['end_time'])
        if start.strftime('%Y-%m-%d') != date:
            continue
        folder = dataset / date / f'{start:%H%M}-{end:%H%M}'
        if folder.is_dir():
            events.append((folder, start, end, state, marker))
    if not events:
        raise ValueError(f'No existing event folders with exact range state for {date}')
    actual = {p for p in (dataset / date).iterdir() if p.is_dir() and len(p.name) == 9}
    if actual != {e[0] for e in events} or len(events) != len(actual):
        raise ValueError('Event folders and range states do not match uniquely')
    return events


def selected_dates(dataset, date=None, month=None):
    """Only existing candidate day folders; calendar days without events are skipped."""
    return sorted(p.name for p in dataset.iterdir() if p.is_dir()
                  and len(p.name) == 10
                  and (p.name == date if date else p.name.startswith(month + '-')))


def input_files(args, date):
    day = pd.Timestamp(date)
    tag = f'{day:%Y%m%d}'
    patterns = [
        (args.raw_root / 'lepe/l2/3dflux' / f'{day:%Y/%m}', f'*_{tag}_v05_*.cdf'),
        (args.raw_root / 'mgf/l2/8sec' / f'{day:%Y/%m}', f'*_{tag}_v*.cdf'),
        (args.fit_root, f'erg-lepe-fitmom-1min-50eV-1000eV-{tag}-000000-{tag}-240000-v2.nc'),
    ]
    files, missing = [], []
    for folder, pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if not matches:
            missing.append(str(folder / pattern))
        else:
            files.append(matches[-1])
    if missing:
        raise FileNotFoundError('Required local input missing: ' + '; '.join(missing))
    return files


def acquire_missing_inputs(args, date):
    """Opt-in, serial acquisition of missing raw products; PI data stay local."""
    day = pd.Timestamp(date)
    tag = f'{day:%Y%m%d}'
    need_lepe = not list((args.raw_root / 'lepe/l2/3dflux' / f'{day:%Y/%m}').glob(f'*_{tag}_v05_*.cdf'))
    need_mgf = not list((args.raw_root / 'mgf/l2/8sec' / f'{day:%Y/%m}').glob(f'*_{tag}_v*.cdf'))
    if not need_lepe and not need_mgf:
        return
    if args.raw_root.parts[-2:] != ('satellite', 'erg'):
        raise ValueError('--download-missing requires --raw-root ending in satellite/erg')
    import ergpyspedas.erg as ergpy
    from ergpyspedas.erg.satellite.erg.config import CONFIG
    # Instrument loaders append satellite/erg/<instrument>/... to this root.
    CONFIG['local_data_dir'] = str(args.raw_root.parent.parent)
    trange = [f'{day:%Y-%m-%d}/00:00:00', f'{day:%Y-%m-%d}/23:59:59']
    if need_lepe:
        print(f'Acquiring missing LEP-e L2 3dflux for {date}', flush=True)
        ergpy.lepe(trange=trange, datatype='3dflux', level='l2', downloadonly=True, ror=False)
    if need_mgf:
        print(f'Acquiring missing MGF L2 8sec for {date}', flush=True)
        ergpy.mgf(trange=trange, datatype='8sec', level='l2', downloadonly=True, ror=False)


def read_var(cdf, name, first=None, last=None):
    attrs = cdf.varattsget(name)
    args = {} if first is None else dict(startrec=int(first), endrec=int(last))
    data = np.array(cdf.varget(name, **args), copy=True)
    if np.issubdtype(data.dtype, np.floating) and 'FILLVAL' in attrs:
        data[data == float(attrs['FILLVAL'])] = np.nan
    return data


def compute_native(cdf, source, btime, b, bquality, start, end, packet_mask, energy_upper=np.inf):
    """Load a small record slice in the public LEP-e loader's array convention."""
    import pyspedas as psp
    from cdflib import cdfepoch
    times = np.asarray(cdfepoch.to_datetime(cdf.varget('epoch'))).astype('datetime64[ns]')
    if np.isnat(times).any() or np.any(np.diff(times).astype('int64') <= 0):
        raise ValueError('LEP-e epochs must be finite and strictly increasing')
    first, stop = np.searchsorted(times, [start.to_datetime64(), end.to_datetime64()])
    if first == stop:
        raise ValueError('No LEP-e records in this event')
    t = times[first:stop]
    unix = t.astype('int64') / 1e9
    flux = read_var(cdf, 'fedu', first, stop - 1).transpose(0, 2, 3, 1)
    energy = read_var(cdf, 'fedu_energy', first, stop - 1)
    packet = read_var(cdf, 'packet_info', first, stop - 1)
    b_interp, bvalid, gaps, left, right = field_at_times(btime, b, unix)
    psp.del_data('*')
    # Matches ergpyspedas.lepe: energy/channel/sector, sector coordinate 0..15.
    psp.store_data(PREFIX, data={'x': unix, 'y': flux, 'v1': energy,
                                 'v2': np.arange(flux.shape[2]), 'v3': np.arange(16)},
                   attr_dict={'CDF': {'FILENAME': str(source), 'VATT': cdf.varattsget('fedu'),
                                      'GATT': cdf.globalattsget()}})
    # Supply a finite field for internal tensor eigensystem diagnostics only at invalid
    # timestamps; all temperatures at these timestamps are rejected below.
    safe_b = b_interp.copy()
    safe_b[~bvalid] = [0., 0., 1.]
    psp.store_data('comparison_mgf_dsi', data={'x': unix, 'y': safe_b})
    names = psp.projects.erg.erg_lep_part_products(
        PREFIX, outputs=['moments'], energy=[67., energy_upper],
        mag_name='comparison_mgf_dsi')
    if not names or PREFIX + '_ttens' not in names:
        raise RuntimeError('Moment calculation did not return a temperature tensor')
    tensor = np.asarray(psp.get_data(PREFIX + '_ttens').y)
    raw_temp = project_temperature(tensor, b_interp)
    np.testing.assert_allclose(raw_temp[:, 0], psp.get_data(PREFIX + '_avgtemp').y,
                               rtol=1e-10, atol=1e-8, equal_nan=True)
    density = np.asarray(psp.get_data(PREFIX + '_density').y)
    valid = (bvalid & ((packet.astype('int64') & packet_mask) == 0)
             & np.isfinite(density) & (density > 0)
             & np.isfinite(raw_temp).all(axis=1) & (raw_temp > 0).all(axis=1))
    temp = raw_temp.copy()
    temp[~valid] = np.nan
    frame = pd.DataFrame(temp, index=pd.DatetimeIndex(t, name='time_utc'),
                         columns=[f'moment_{k}_eV' for k in COMPONENTS])
    frame['valid'] = valid
    frame['packet_info_raw'] = packet
    frame['magnetic_valid'] = bvalid
    frame['magnetic_bracket_gap_s'] = gaps
    frame['density_cm3'] = density
    selected = energy[np.isfinite(energy) & (energy >= 67) & (energy <= energy_upper)]
    diagnostics = dict(time_utc=t, ttens_dsi_eV=tensor, b_dsi_nT=b_interp,
                       raw_temperature_eV=raw_temp, packet_info_raw=packet,
                       mgf_quality_left_raw=bquality[left], mgf_quality_right_raw=bquality[right])
    return frame, diagnostics, [float(np.nanmin(selected)), float(np.nanmax(selected))]


def minute_comparison(native, fit, lo, hi, flux_threshold, temperature_max):
    index = pd.date_range(lo, hi, freq='1min', inclusive='left', name='time_utc')
    result = native[[f'moment_{k}_eV' for k in COMPONENTS]].resample('1min').mean().reindex(index)
    result['moment_n_total'] = native.valid.resample('1min').size().reindex(index, fill_value=0)
    result['moment_n_valid'] = native.valid.resample('1min').sum().reindex(index, fill_value=0)
    result['moment_valid_fraction'] = result.moment_n_valid / result.moment_n_total.replace(0, np.nan)
    fit = fit.reindex(time=index)
    parallel, perpendicular = fit.T_para.values, fit.T_perp.values
    good_p = (np.isfinite(parallel) & (parallel > 0)
              & (fit.eflux_para_mean.values > flux_threshold) & (fit.num_valid_data.values > 3))
    good_t = (np.isfinite(perpendicular) & (perpendicular > 0)
              & (fit.eflux_perp_mean.values > flux_threshold) & (fit.num_valid_data.values > 3))
    if temperature_max is not None:
        good_p &= parallel < temperature_max
        good_t &= perpendicular < temperature_max
    for k, raw, good in zip(COMPONENTS, [(parallel + 2 * perpendicular) / 3, parallel, perpendicular],
                             [good_p & good_t, good_p, good_t]):
        result[f'fit_{k}_raw_eV'] = raw
        result[f'fit_{k}_valid'] = good
        result[f'fit_{k}_eV'] = np.where(good, raw, np.nan)
        result[f'moment_over_fit_{k}'] = result[f'moment_{k}_eV'] / result[f'fit_{k}_eV']
    for k in ['log_residual', 'num_valid_data', 'eflux_para_mean', 'eflux_perp_mean']:
        result[f'fit_{k}'] = fit[k].values
    return result


def combine_profiles(natives, tables):
    """Keep independent integral quality counts and a single shared PI-fit series."""
    first = next(iter(tables.values()))
    combined = first[[k for k in first if k.startswith('fit_')]].copy()
    native_parts = []
    for profile, table in tables.items():
        pd.testing.assert_frame_equal(combined[[k for k in combined if k.startswith('fit_')]],
                                      table[[k for k in table if k.startswith('fit_')]])
        columns = {k: k.replace('moment_', f'moment_{profile}_', 1)
                   for k in table if k.startswith('moment_')}
        combined = combined.join(table[list(columns)].rename(columns=columns), how='outer')
        native_parts.append(natives[profile].add_prefix(profile + '__'))
    native_combined = pd.concat(native_parts, axis=1, join='outer')
    return native_combined, combined


def result_signature(source_meta, marker, args, start, end):
    payload = dict(sources=source_meta, range_state_sha256=sha256(marker),
                   script_sha256=sha256(__file__), start=str(start), end=str(end),
                   profiles={k: [67., None if np.isinf(v['upper']) else v['upper']]
                             for k, v in PROFILES.items()},
                   fit_flux_min=args.fit_flux_min, fit_temperature_max=args.fit_temperature_max,
                   packages={k: version(k) for k in ['pyspedas', 'ergpyspedas', 'cdflib', 'numpy', 'xarray']})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reusable_result(target, signature):
    marker = target / f'{STEM}_provenance.json'
    try:
        metadata = json.loads(marker.read_text())
        expected = {STEM + suffix for suffix in OUTPUT_SUFFIXES}
        checksums = metadata['output_sha256']
        if metadata['signature'] != signature or set(checksums) != expected:
            return None
        if any(sha256(target / name) != digest for name, digest in checksums.items()):
            return None
        return metadata['summary']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def plot_event(natives, table, start, end, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout='constrained')
    for ax, key, title in zip(axes, COMPONENTS, [r'$T_{\rm eff}=(T_\parallel+2T_\perp)/3$',
                                               r'$T_\parallel$', r'$T_\perp$']):
        center = table.index + pd.Timedelta(seconds=30)
        for profile, config in PROFILES.items():
            native = natives[profile]
            ax.plot(native.index, native[f'moment_{key}_eV'], color=config['color'], alpha=.25, lw=.7)
            ax.plot(center, table[f'moment_{profile}_{key}_eV'], 'o-',
                    color=config['color'], ms=3, label=config['label'] + ' (1-min mean)')
        ax.errorbar(center, table[f'fit_{key}_eV'], xerr=pd.Timedelta(seconds=30),
                    fmt='s-', color='C1', ms=4, lw=1, label='PI fit: fit to 1-min mean distribution')
        bad = ~table[f'fit_{key}_valid'] & (table[f'fit_{key}_raw_eV'] > 0)
        if bad.any():
            ax.plot(center[bad], table.loc[bad, f'fit_{key}_raw_eV'], 'x', color='.5',
                    label='PI fit: rejected by quality criteria')
        ax.axvspan(table.index[0], start, color='.8', alpha=.3)
        ax.axvspan(end, table.index[-1] + pd.Timedelta(minutes=1), color='.8', alpha=.3)
        ax.set(yscale='log', ylabel=title + ' [eV]')
        ax.grid(alpha=.25, which='both')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set_title(f'LEP-e | {start:%Y-%m-%d %H:%M:%S} - {end:%H:%M:%S} UTC\n'
                      'Integral: E >= 67 eV / 67-1000 eV | PI fit: 50-1000 eV')
    axes[-1].set_xlim(table.index[0], table.index[-1] + pd.Timedelta(minutes=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('UTC (1-min points at interval centers; horizontal bars show fit intervals)')
    for ext in ('png', 'pdf'):
        fig.savefig(target / f'{STEM}.{ext}', dpi=170)
    plt.close(fig)



def output_day(args, date):
    return (args.output_root / args.dataset.name / date) if args.output_root else args.dataset / date


def check_events(events, date):
    boundary = pd.Timestamp(date) + pd.Timedelta(days=1)
    for folder, start, end, _, _ in events:
        if start >= end or end.ceil('min') > boundary:
            raise ValueError(f'Invalid/cross-day event unsupported: {folder}')


def run_day(args):
    events = discover_events(args.dataset, args.date)
    check_events(events, args.date)
    if args.download_missing and not args.dry_run:
        acquire_missing_inputs(args, args.date)
    sources = input_files(args, args.date)
    print(f'{args.date}: {len(events)} events; inputs: ' + ', '.join(str(p) for p in sources), flush=True)
    if args.dry_run:
        return 0
    source_meta = [{'path': str(p), 'sha256': sha256(p)} for p in sources]
    summary, pending = [], []
    out_day = output_day(args, args.date)
    out_day.mkdir(parents=True, exist_ok=True)
    summary_dir = out_day / OUTPUT_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'{STEM}_summary.csv'
    for event in events:
        folder, start, end, state, marker = event
        target = out_day / folder.name / OUTPUT_DIR
        signature = result_signature(source_meta, marker, args, start, end)
        prior = None if args.force else reusable_result(target, signature)
        if prior:
            summary.append({**prior, 'status': 'skipped'})
            print(f'Skip verified {folder.name}', flush=True)
        else:
            pending.append((event, target, signature))
    if not pending:
        pd.DataFrame(summary).sort_values('start_utc').to_csv(summary_path, index=False)
        return 0
    from cdflib import CDF, cdfepoch
    import xarray as xr
    with xr.open_dataset(sources[2]) as fit_ds:
        fit = fit_ds.load()
    if fit.attrs['resampling'] != '1min' or float(fit.attrs['energy_min']) != 50 or float(fit.attrs['energy_max']) != 1000:
        raise ValueError('Unexpected PI fit specification')
    fit_energy = fit.energy.values[(fit.energy.values >= 50) & (fit.energy.values <= 1000)]
    with staged_cdf(sources[0], args.scratch) as raw_path, staged_cdf(sources[1], args.scratch) as mag_path:
        raw, mag = CDF(raw_path), CDF(mag_path)
        btime = np.asarray(cdfepoch.to_datetime(mag.varget('epoch_8sec'))).astype('datetime64[ns]').astype('int64') / 1e9
        b, bq = read_var(mag, 'mag_8sec_dsi'), read_var(mag, 'quality_8sec')
        for (folder, start, end, state, marker), target, signature in pending:
            row = dict(event=folder.name, start_utc=str(start), end_utc=str(end), status='failed')
            try:
                target.mkdir(parents=True, exist_ok=True)
                (target / f'{STEM}_provenance.json').unlink(missing_ok=True)
                print(f'Computing {folder.name} (both energy ranges)', flush=True)
                lo, hi = start.floor('min'), end.ceil('min')
                natives, tables, diagnostics, bounds, counts = {}, {}, {}, {}, {}
                for profile, config in PROFILES.items():
                    native, diag, energy_bounds = compute_native(
                        raw, raw_path, btime, b, bq, lo, hi,
                        int(fit.attrs['packet_info_bitmask']), energy_upper=config['upper'])
                    natives[profile] = native
                    tables[profile] = minute_comparison(native, fit, lo, hi, args.fit_flux_min, args.fit_temperature_max)
                    diagnostics.update({profile + '__' + k: v for k, v in diag.items()})
                    bounds[profile] = energy_bounds
                    counts[profile] = dict(native_total=len(native), native_valid=int(native.valid.sum()))
                    row[f'{profile}_native_total'] = len(native)
                    row[f'{profile}_native_valid'] = int(native.valid.sum())
                    for key in COMPONENTS:
                        ratio = tables[profile][f'moment_over_fit_{key}'].dropna()
                        row[f'{profile}_{key}_n_paired_minutes'] = len(ratio)
                        row[f'{profile}_{key}_median_moment_over_fit'] = float(ratio.median()) if len(ratio) else np.nan
                native_combined, table = combine_profiles(natives, tables)
                plot_event(natives, table, start, end, target)
                native_combined.to_csv(target / f'{STEM}_native.csv')
                table.to_csv(target / f'{STEM}_1min.csv')
                np.savez_compressed(target / f'{STEM}_diagnostics.npz', **diagnostics)
                row['status'] = 'complete'
                metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), signature=signature,
                                event_start_utc=str(start), event_end_utc=str(end), original_event_folder=str(folder),
                                range_id=state['range_id'], batch_status=state['status'],
                                range_state_path=str(marker), range_state_sha256=sha256(marker),
                                sources=source_meta, script_sha256=sha256(__file__),
                                packages={k: version(k) for k in ['pyspedas', 'ergpyspedas', 'cdflib', 'numpy', 'xarray']},
                                moment_energy_center_ranges_eV={'all': [67., None], 'le1000': [67., 1000.]},
                                moment_actual_energy_center_bounds_eV=bounds,
                                fit_actual_energy_centers_eV=fit_energy,
                                energy_policy='Integral: E >= 67 eV, all measured channels; and 67 <= E <= 1000 eV, whole bins by center. PI fit: 50 <= E <= 1000 eV',
                                effective_definition='(T_parallel + 2*T_perpendicular)/3 = trace(T_DSI)/3',
                                projection='T_parallel=bhat^T T_DSI bhat; T_perpendicular=(trace(T_DSI)-T_parallel)/2',
                                field='MGF L2 8sec DSI, linear interpolation at native LEP-e spin-start epoch; max bracket gap 16 s; no extrapolation',
                                field_quality='MGF quality flags retained in diagnostics; no additional flag-based selection',
                                spacecraft_potential_V=0.,
                                native_quality='PI packet_info bitmask applied; positive density and all temperatures; finite field/tensor; independent masks for each energy range',
                                packet_info_bitmask=int(fit.attrs['packet_info_bitmask']),
                                fit_quality=dict(flux_strict_min=args.fit_flux_min, num_valid_data_strict_min=3,
                                                 temperature_strict_max_eV=args.fit_temperature_max,
                                                 residual_threshold=None, effective_requires_both_components=True),
                                fit_attributes=fit.attrs,
                                time_policy='UTC; whole [minute, minute+1min) intervals overlapping event, including outside-event portions; plot at centers; exact event bounds shaded',
                                averaging='Arithmetic mean of valid native temperatures, NOT moments of a 1min-averaged distribution; PI fits the 1min-averaged distribution',
                                coverage='Counts and valid fractions saved for each energy range; no minimum 1min coverage threshold imposed',
                                counts=counts, summary=row,
                                output_sha256={STEM + suffix: sha256(target / (STEM + suffix)) for suffix in OUTPUT_SUFFIXES})
                (target / f'{STEM}_provenance.json').write_text(json.dumps(metadata, indent=2, default=json_default) + '\n')
            except Exception as exc:
                row['status'] = 'failed'
                row['error'] = f'{type(exc).__name__}: {exc}'
                traceback.print_exc()
            summary.append(row)
            pd.DataFrame(summary).sort_values('start_utc').to_csv(summary_path, index=False)
            print(json.dumps(row), flush=True)
    return int(any(row['status'] == 'failed' for row in summary))


def run_month(args, dates):
    inventory = []
    for date in dates:
        events = discover_events(args.dataset, date)
        row = dict(date=date, n_events=len(events), status='ready')
        try:
            check_events(events, date)
            files = input_files(args, date)
            row['inputs'] = [str(p) for p in files]
        except (ValueError, FileNotFoundError) as exc:
            row.update(status='missing_or_unsupported_input', error=str(exc))
        inventory.append(row)
        print(json.dumps(row), flush=True)
    print(f"{args.month}: {len(dates)} event days, {sum(r['n_events'] for r in inventory)} events, "
          f"{sum(r['status'] != 'ready' for r in inventory)} days with missing/unsupported input", flush=True)
    if args.dry_run:
        return int(any(r['status'] != 'ready' for r in inventory))
    root = (args.output_root / args.dataset.name) if args.output_root else args.dataset
    summary_dir = root / OUTPUT_DIR
    logs = summary_dir / 'logs' / args.month
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for info in inventory:
        date = info['date']
        log_path = logs / (date + '.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--date', date,
                   '--dataset', str(args.dataset), '--raw-root', str(args.raw_root),
                   '--fit-root', str(args.fit_root), '--scratch', str(args.scratch),
                   '--fit-flux-min', str(args.fit_flux_min)]
        if args.output_root:
            command.extend(['--output-root', str(args.output_root)])
        if args.fit_temperature_max is not None:
            command.extend(['--fit-temperature-max', str(args.fit_temperature_max)])
        if args.force:
            command.append('--force')
        if args.download_missing:
            command.append('--download-missing')
        print(f'Processing {date} ({info["n_events"]} events); log: {log_path}', flush=True)
        with log_path.open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        rows.append(dict(date=date, n_events=info['n_events'],
                         status='complete' if result.returncode == 0 else 'failed',
                         returncode=result.returncode, log=str(log_path)))
        pd.DataFrame(rows).to_csv(summary_dir / f'{STEM}_month_{args.month.replace("-", "")}_summary.csv', index=False)
        print(json.dumps(rows[-1]), flush=True)
    return int(any(row['returncode'] for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument('--date', help='UTC date YYYY-MM-DD')
    period.add_argument('--month', help='UTC month YYYY-MM or YYYYMM')
    parser.add_argument('--dataset', type=Path, help='Existing auto/<dataset key> directory')
    parser.add_argument('--output-root', type=Path, help='Staging root; otherwise save in each event/lepe_temperature_comparison/')
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--fit-root', type=Path, default=FIT_ROOT)
    parser.add_argument('--scratch', type=Path, default=Path('/tmp'))
    parser.add_argument('--fit-flux-min', type=float, default=1e4)
    parser.add_argument('--fit-temperature-max', type=float, default=None,
                        help='Optional strict upper quality threshold [eV]; none by default')
    parser.add_argument('--dry-run', action='store_true', help='List events and local inputs only; no output writes')
    parser.add_argument('--force', action='store_true', help='Recompute even verified complete comparisons')
    parser.add_argument('--download-missing', action='store_true',
                        help='Acquire missing LEP-e/MGF CDFs serially; existing files are reused; PI fit must be local')
    args = parser.parse_args(argv)
    try:
        if args.date:
            if pd.Timestamp(args.date).strftime('%Y-%m-%d') != args.date:
                raise ValueError('Use YYYY-MM-DD')
        else:
            if len(args.month) == 6 and args.month.isdigit():
                args.month = args.month[:4] + '-' + args.month[4:]
            if pd.Timestamp(args.month + '-01').strftime('%Y-%m') != args.month:
                raise ValueError('Use YYYY-MM or YYYYMM')
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.fit_flux_min < 0 or not np.isfinite(args.fit_flux_min):
        parser.error('fit-flux-min must be finite and nonnegative')
    if args.fit_temperature_max is not None and (not np.isfinite(args.fit_temperature_max) or args.fit_temperature_max <= 0):
        parser.error('fit-temperature-max must be positive and finite')
    if args.dataset is None:
        candidates = [p for p in (ANALYSIS_ROOT / 'auto').iterdir()
                      if p.is_dir() and selected_dates(p, args.date, args.month)]
        if len(candidates) != 1:
            parser.error(f'Expected one matching dataset, found {len(candidates)}; specify --dataset')
        args.dataset = candidates[0]
    dates = selected_dates(args.dataset, args.date, args.month)
    if not dates:
        parser.error('No existing candidate day folders in the requested period')
    return run_month(args, dates) if args.month else run_day(args)


if __name__ == '__main__':
    raise SystemExit(main())
