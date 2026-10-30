"""PI multiband v3: native full-range and minute low/high temperature comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

import compare as core
import numpy as np
import pandas as pd

VERSION = 'pi_multiband_v3'
FIT_BASE = Path('/mnt/j/observation_data/lepe-fitmom-data-v2')
BANDS = {
    'all': dict(lower=67., upper=np.inf, fit_min=50., fit_max=20000., resolution='raw'),
    'low': dict(lower=67., upper=1000., fit_min=50., fit_max=1000., resolution='1min'),
    'high': dict(lower=1000., upper=20000., fit_min=1000., fit_max=20000., resolution='1min'),
}
FIT_VARIABLES = ('ne', 'T_para', 'T_perp', 'log_residual', 'num_valid_data',
                 'eflux_para_mean', 'eflux_perp_mean')
OUTPUTS = ('temperature_comparison.png', 'temperature_comparison.pdf',
           'native_comparison.csv', 'minute_comparison.csv', 'diagnostics.npz')
PROVENANCE = 'provenance.json'
PACKAGES = ('pyspedas', 'ergpyspedas', 'cdflib', 'numpy', 'pandas', 'xarray', 'matplotlib')


def code_hashes():
    return {p.name: core.sha256(p) for p in [Path(__file__), Path(core.__file__)]}


def options(args):
    return {k: getattr(args, k) for k in (
        'low_fit_flux_min', 'high_fit_flux_min', 'raw_fit_flux_min',
        'fit_temperature_max', 'raw_match_tolerance_ms', 'plot_temperature_max')}


def input_files(args, date):
    day = pd.Timestamp(date)
    tag = f'{day:%Y%m%d}'
    paths = core.input_files(SimpleNamespace(raw_root=args.raw_root,
                                             fit_root=args.fit_1min_root / f'{day:%Y}'), date)
    for root, resolution, lo, hi in [(args.fit_1min_root, '1min', 1000, 20000),
                                     (args.fit_raw_root, 'raw', 50, 20000)]:
        path = root / f'{day:%Y}' / f'erg-lepe-fitmom-{resolution}-{lo}eV-{hi}eV-{tag}-000000-{tag}-240000-v2.nc'
        if not path.is_file():
            raise FileNotFoundError(f'Required PI input missing: {path}')
        paths.append(path)
    return dict(zip(('lepe', 'mgf', 'low', 'high', 'all'), paths))


def load_fit(path, band):
    import xarray as xr
    config = BANDS[band]
    with xr.open_dataset(path) as ds:
        if (str(ds.attrs.get('fitmom_version')) != '2'
                or ds.attrs.get('resampling') != config['resolution']
                or float(ds.attrs.get('energy_min', np.nan)) != config['fit_min']
                or float(ds.attrs.get('energy_max', np.nan)) != config['fit_max']):
            raise ValueError(f'Unexpected PI fit specification: {path}')
        t = ds.time.values.astype('datetime64[ns]')
        if not len(t) or np.isnat(t).any() or np.any(np.diff(t).astype('int64') <= 0):
            raise ValueError(f'PI timestamps must be strictly increasing: {path}')
        if config['resolution'] == '1min' and np.any(t != t.astype('datetime64[m]')):
            raise ValueError(f'PI minute timestamps must be minute boundaries: {path}')
        frame = ds[list(FIT_VARIABLES)].to_dataframe().copy()
        frame.index = pd.DatetimeIndex(t, name='time_utc')
        energies = ds.energy.values
        selected = energies[(energies >= config['fit_min']) & (energies <= config['fit_max'])]
        meta = dict(attributes=dict(ds.attrs), actual_energy_centers_eV=selected,
                    time_encoding={k: ds.time.encoding.get(k) for k in ('units', 'calendar')},
                    time_first=str(t[0]), time_last=str(t[-1]), n_records=len(t),
                    units_source='PI accompanying documentation and generation code; variable unit attributes are absent')
    return frame, meta


def fit_quality(frame, flux_min=None, temperature_max=None):
    """Bit reasons: 1 invalid temperature, 2 <=3 data points, 4 flux, 8 upper cut."""
    out = frame.copy()
    valid_count = np.isfinite(frame.num_valid_data) & (frame.num_valid_data > 3)
    reasons = {}
    for key, temp, flux in [('parallel', 'T_para', 'eflux_para_mean'),
                            ('perpendicular', 'T_perp', 'eflux_perp_mean')]:
        value = frame[temp].to_numpy(dtype=float)
        reason = np.zeros(len(frame), dtype=np.uint8)
        reason[~(np.isfinite(value) & (value > 0))] |= 1
        reason[~valid_count.to_numpy()] |= 2
        if flux_min is not None:
            reason[~(np.isfinite(frame[flux]) & (frame[flux] > flux_min)).to_numpy()] |= 4
        if temperature_max is not None:
            reason[value >= temperature_max] |= 8
        reasons[key] = reason
        out[f'{key}_raw_eV'] = value
        out[f'{key}_valid'] = reason == 0
        out[f'{key}_reason'] = reason
        out[f'{key}_eV'] = np.where(reason == 0, value, np.nan)
        out[f'{key}_nonpositive_flux_warning'] = ~(np.isfinite(frame[flux]) & (frame[flux] > 0))
    reason = reasons['parallel'] | reasons['perpendicular']
    value = (out.parallel_raw_eV + 2 * out.perpendicular_raw_eV) / 3
    out['effective_raw_eV'] = value
    out['effective_valid'] = reason == 0
    out['effective_reason'] = reason
    out['effective_eV'] = value.where(reason == 0)
    return out


def component_native(native, diagnostic, packet_mask):
    """Retain independent component masks, with common packet/field/density checks."""
    result = native.copy()
    common = (native.magnetic_valid.to_numpy()
              & ((native.packet_info_raw.to_numpy(dtype='int64') & packet_mask) == 0)
              & np.isfinite(native.density_cm3) & (native.density_cm3 > 0))
    for i, key in enumerate(core.COMPONENTS):
        raw = diagnostic['raw_temperature_eV'][:, i]
        valid = common & np.isfinite(raw) & (raw > 0)
        result[f'moment_{key}_raw_eV'] = raw
        result[f'moment_{key}_eV'] = np.where(valid, raw, np.nan)
        result[f'{key}_valid'] = valid
    result['valid'] = result[[f'{k}_valid' for k in core.COMPONENTS]].all(axis=1)
    return result


def minute_table(native, fit, lo, hi):
    index = pd.date_range(lo, hi, freq='1min', inclusive='left', name='time_utc')
    out = fit.reindex(index).add_prefix('fit_')
    out['moment_n_total'] = native.valid.resample('1min').size().reindex(index, fill_value=0)
    for key in core.COMPONENTS:
        out[f'moment_{key}_eV'] = native[f'moment_{key}_eV'].resample('1min').mean().reindex(index)
        count = native[f'{key}_valid'].resample('1min').sum().reindex(index, fill_value=0)
        out[f'moment_{key}_n_valid'] = count
        out[f'moment_{key}_valid_fraction'] = count / out.moment_n_total.replace(0, np.nan)
        out[f'moment_over_fit_{key}'] = out[f'moment_{key}_eV'] / out[f'fit_{key}_eV']
    return out


def match_raw(native_index, fit, tolerance_ms=1.):
    """Nearest, within tolerance, one-to-one. Reject all competing matches."""
    targets = pd.DatetimeIndex(native_index).to_numpy(dtype='datetime64[ns]').astype('int64')
    sources = fit.index.to_numpy(dtype='datetime64[ns]').astype('int64')
    out = pd.DataFrame(index=pd.DatetimeIndex(native_index, name='time_utc'))
    if len(sources):
        if np.any(np.diff(sources) <= 0):
            raise ValueError('PI raw timestamps must be unique and increasing')
        right = np.searchsorted(sources, targets).clip(0, len(sources) - 1)
        left = np.maximum(right - 1, 0)
        best = np.where(abs(targets - sources[left]) <= abs(targets - sources[right]), left, right)
        delta = sources[best] - targets
        within = abs(delta) <= round(tolerance_ms * 1e6)
        counts = np.bincount(best[within], minlength=len(sources))
        ambiguous = within & (counts[best] > 1)
        valid = within & ~ambiguous
        out['nearest_offset_s'] = delta / 1e9
        out['status'] = np.where(ambiguous, 'ambiguous', np.where(valid, 'matched', 'outside_tolerance'))
        out['pi_source_index'] = np.where(valid, best, -1)
        out['pi_time_utc'] = pd.to_datetime(np.where(valid, sources[best], np.iinfo('int64').min))
        matched = fit.iloc[best].copy()
        matched.index = out.index
        for name in matched:
            out['fit_' + name] = matched[name].where(valid)
    else:
        out['nearest_offset_s'] = np.nan
        out['status'] = 'no_raw_samples'
        out['pi_source_index'] = -1
        out['pi_time_utc'] = pd.NaT
        for name in fit:
            out['fit_' + name] = np.nan
    return out


def signature(inputs, marker, args, start, end):
    payload = dict(version=VERSION, inputs=inputs, range_state_sha256=core.sha256(marker),
                   code=code_hashes(), options=options(args), start=str(start), end=str(end),
                   packages={p: core.version(p) for p in PACKAGES})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reusable(target, expected):
    try:
        meta = json.loads((target / PROVENANCE).read_text())
        if meta['signature'] != expected or set(meta['output_sha256']) != set(OUTPUTS):
            return None
        if any(core.sha256(target / name) != digest for name, digest in meta['output_sha256'].items()):
            return None
        return meta['summary']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def draw_series(ax, times, values, *, cap=None, label=None, color='C0', marker='.', minute=False, **kw):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values) & (values > 0)
    above = good & (values > cap) if cap is not None else np.zeros(len(values), dtype=bool)
    shown = np.where(good & ~above, values, np.nan)
    if minute:
        ax.errorbar(times, shown, xerr=pd.Timedelta(seconds=30), fmt=marker,
                    color=color, label=label, **kw)
    else:
        ax.plot(times, shown, marker=marker, color=color, label=label, **kw)
    if above.any():
        ax.plot(pd.DatetimeIndex(times)[above], np.full(above.sum(), cap), '^', color=color, ms=5)
        return int(above.sum()), float(values[above].max())
    return 0, None


def plot_event(natives, fits, minutes, start, end, lo, hi, target, cap=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex=True, layout='constrained')
    names = {'all': 'Full: Integral E >= 67 eV | PI raw 50-20000 eV',
             'low': 'Low: Integral 67-1000 eV | PI 1min 50-1000 eV',
             'high': 'High: Integral/PI 1000-20000 eV'}
    labels = [r'$T_{\rm eff}$ [eV]', r'$T_\parallel$ [eV]', r'$T_\perp$ [eV]']
    for col, band in enumerate(BANDS):
        n, f = natives[band], fits[band]
        for row, key in enumerate(core.COMPONENTS):
            ax = axes[row, col]
            off = []
            off.append(draw_series(ax, n.index, n[f'moment_{key}_eV'], cap=cap, color='C0',
                                   label='Integral: native', linestyle='None', ms=2,
                                   alpha=.7 if band == 'all' else .25))
            if band == 'all':
                fit_time = f.index
                off.append(draw_series(ax, fit_time, f[f'{key}_eV'], cap=cap, color='C1',
                                       label='PI fit: raw samples', linestyle='None', ms=2.5))
            else:
                m = minutes[band]
                off.append(draw_series(ax, m.index + pd.Timedelta(seconds=30), m[f'moment_{key}_eV'],
                                       cap=cap, color='C0', label='Integral: mean of native T',
                                       marker='o', linestyle='-', lw=1, ms=3))
                fit_time = f.index + pd.Timedelta(seconds=30)
                off.append(draw_series(ax, fit_time, f[f'{key}_eV'], cap=cap, color='C1',
                                       label='PI fit: fit to mean distribution', marker='s',
                                       minute=True, ms=3, lw=.8))
            rejected = ~f[f'{key}_valid']
            if rejected.any():
                off.append(draw_series(ax, fit_time[rejected], f.loc[rejected, f'{key}_raw_eV'],
                                       cap=cap, color='.5', marker='x', linestyle='None', ms=3,
                                       label='PI fit: rejected'))
            ax.axvspan(lo, start, color='.8', alpha=.3)
            ax.axvspan(end, hi, color='.8', alpha=.3)
            ax.set(yscale='log', xlim=(lo, hi), ylabel=labels[row])
            if cap is not None and ax.get_ylim()[1] > cap * 1.15:
                ax.set_ylim(top=cap * 1.15)
            clipped = [(count, maximum) for count, maximum in off if count]
            if clipped:
                ax.text(.02, .98, f'Above display limit: {sum(c for c, _ in clipped)}; max={max(v for _, v in clipped):.2g} eV',
                        transform=ax.transAxes, va='top', fontsize=7)
            ax.grid(alpha=.2, which='both')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            if row == 0:
                ax.set_title(names[band], fontsize=10)
            if row == 2:
                ax.set_xlabel('UTC')
        axes[1, col].legend(fontsize=7, loc='best')
    fig.suptitle(f'LEP-e | {start:%Y-%m-%d %H:%M:%S} - {end:%H:%M:%S} UTC\n'
                 r'$T_{\rm eff}=(T_\parallel+2T_\perp)/3$; horizontal bars: minute intervals, not temperature errors', fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(target / f'temperature_comparison.{ext}', dpi=160)
    plt.close(fig)


def output_day(args, date):
    base = args.output_root / args.dataset.name if args.output_root else args.dataset
    return base / date


def run_day(args):
    events = core.discover_events(args.dataset, args.date)
    core.check_events(events, args.date)
    if args.download_missing and not args.dry_run:
        core.acquire_missing_inputs(args, args.date)
    paths = input_files(args, args.date)
    print(f'{args.date}: {len(events)} events; ' + ', '.join(f'{k}={v}' for k, v in paths.items()), flush=True)
    if args.dry_run:
        return 0
    inputs = {k: dict(path=str(p), sha256=core.sha256(p)) for k, p in paths.items()}
    rows, pending = [], []
    summary_dir = output_day(args, args.date) / core.OUTPUT_DIR / VERSION
    summary_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        folder, start, end, _, marker = event
        target = output_day(args, args.date) / folder.name / core.OUTPUT_DIR / VERSION
        stamp = signature(inputs, marker, args, start, end)
        old = None if args.force else reusable(target, stamp)
        if old:
            rows.append({**old, 'status': 'skipped'})
            print(f'Skip verified {folder.name}', flush=True)
        else:
            pending.append((event, target, stamp))
    def save_summary():
        pd.DataFrame(rows).sort_values('start_utc').to_csv(summary_dir / 'event_summary.csv', index=False)
    if not pending:
        save_summary()
        return 0
    fit_data, fit_meta = {}, {}
    for band in BANDS:
        frame, fit_meta[band] = load_fit(paths[band], band)
        threshold = getattr(args, {'all': 'raw_fit_flux_min', 'low': 'low_fit_flux_min', 'high': 'high_fit_flux_min'}[band])
        fit_data[band] = fit_quality(frame, threshold, args.fit_temperature_max)
    from cdflib import CDF, cdfepoch
    with core.staged_cdf(paths['lepe'], args.scratch) as lp, core.staged_cdf(paths['mgf'], args.scratch) as mp:
        lepe, mag = CDF(lp), CDF(mp)
        btime = np.asarray(cdfepoch.to_datetime(mag.varget('epoch_8sec'))).astype('datetime64[ns]').astype('int64') / 1e9
        b, bq = core.read_var(mag, 'mag_8sec_dsi'), core.read_var(mag, 'quality_8sec')
        for (folder, start, end, state, marker), target, stamp in pending:
            row = dict(event=folder.name, start_utc=str(start), end_utc=str(end), status='failed')
            try:
                target.mkdir(parents=True, exist_ok=True)
                (target / PROVENANCE).unlink(missing_ok=True)
                print(f'Computing {folder.name}: full / low / high', flush=True)
                lo, hi = start.floor('min'), end.ceil('min')
                natives, fits, minutes, bounds, diagnostic = {}, {}, {}, {}, {}
                for band, config in BANDS.items():
                    packet_mask = int(fit_meta[band]['attributes']['packet_info_bitmask'])
                    n, diag, bounds[band] = core.compute_native(
                        lepe, lp, btime, b, bq, lo, hi, packet_mask,
                        energy_lower=config['lower'], energy_upper=config['upper'])
                    n = component_native(n, diag, packet_mask)
                    natives[band] = n
                    diagnostic.update({band + '__' + k: v for k, v in diag.items()})
                    pad = pd.Timedelta(milliseconds=args.raw_match_tolerance_ms) if band == 'all' else pd.Timedelta(0)
                    f = fit_data[band]
                    f = f[(f.index >= lo - pad) & (f.index < hi + pad)].copy()
                    fits[band] = f
                    if band != 'all':
                        minutes[band] = minute_table(n, f, lo, hi)
                    for key in core.COMPONENTS:
                        row[f'{band}_{key}_n_native_valid'] = int(n[f'{key}_valid'].sum())
                native_csv = pd.concat([n.add_prefix(band + '__') for band, n in natives.items()], axis=1)
                raw = fits['all']
                match = match_raw(natives['all'].index, raw, args.raw_match_tolerance_ms)
                matched_in_event = (match.index >= start) & (match.index < end)
                row['raw_n_matched_in_event'] = int(((match.status == 'matched') & matched_in_event).sum())
                row['raw_n_unmatched_in_event'] = int(((match.status != 'matched') & matched_in_event).sum())
                for key in core.COMPONENTS:
                    match[f'moment_over_fit_{key}'] = natives['all'][f'moment_{key}_eV'] / match[f'fit_{key}_eV']
                    for band in BANDS:
                        ratios = (match.loc[matched_in_event, f'moment_over_fit_{key}'] if band == 'all'
                                  else minutes[band][f'moment_over_fit_{key}']).dropna()
                        row[f'{band}_{key}_n_paired'] = len(ratios)
                        row[f'{band}_{key}_median_moment_over_fit'] = float(ratios.median()) if len(ratios) else np.nan
                native_csv = native_csv.join(match.add_prefix('raw_match__'), how='outer')
                native_csv = native_csv.join(raw.add_prefix('pi_raw__'), how='outer').sort_index()
                native_csv['record_has_integral'] = native_csv.index.isin(natives['all'].index)
                native_csv['record_has_pi_raw'] = native_csv.index.isin(raw.index)
                native_csv['within_event'] = (native_csv.index >= start) & (native_csv.index < end)
                minute_csv = pd.concat([frame.add_prefix(band + '__') for band, frame in minutes.items()], axis=1)
                for band, f in fits.items():
                    diagnostic[band + '__pi_time_utc'] = f.index.to_numpy(dtype='datetime64[ns]')
                    for name in f:
                        diagnostic[band + '__pi_' + name] = f[name].to_numpy()
                plot_event(natives, fits, minutes, start, end, lo, hi, target, args.plot_temperature_max)
                native_csv.to_csv(target / 'native_comparison.csv', index_label='time_utc')
                minute_csv.to_csv(target / 'minute_comparison.csv', index_label='time_utc')
                np.savez_compressed(target / 'diagnostics.npz', **diagnostic)
                row['status'] = 'complete'
                meta = dict(version=VERSION, signature=stamp, created_utc=datetime.now(timezone.utc).isoformat(),
                            event_start_utc=str(start), event_end_utc=str(end), original_event_folder=str(folder),
                            range_id=state['range_id'], batch_status=state['status'], inputs=inputs,
                            range_state_path=str(marker), range_state_sha256=core.sha256(marker),
                            code_sha256=code_hashes(), packages={p: core.version(p) for p in PACKAGES}, options=options(args),
                            profiles={k: dict(integral_center_range_eV=[v['lower'], None if np.isinf(v['upper']) else v['upper']],
                                              integral_actual_centers_bounds_eV=bounds[k], pi=fit_meta[k]) for k, v in BANDS.items()},
                            field='MGF L2 8sec DSI, linear interpolation at native spin-start epochs; max gap 16 s; no extrapolation; raw MGF quality retained without additional flag selection',
                            integral='Public erg_lep_part_products, central tensor; full native energy-bin widths selected by center; spacecraft potential 0 V; component-specific validity',
                            projection='Tpara=bhat^T T_DSI bhat; Tperp=(trace(T_DSI)-Tpara)/2; Teff=trace(T_DSI)/3; all eV',
                            timing='UTC. PI raw uses actual nonuniform timestamps, no assumed +/-4 s support. Native ratios use nearest one-to-one matching within tolerance; ambiguous matches rejected. Low/high minute ratios use complete overlapping UTC minute bins, including outside-event portions; plotted at centers with +/-30 s support.',
                            averaging='Low/high Integral: mean of valid native temperatures; PI: fit to mean distribution. Full: native Integral vs native PI fit, no minute averaging.',
                            quality='Independent per dataset/component. Effective PI requires both components. num_valid_data>3 is only a minimal point-count condition. Rejection bits 1=invalid T,2=insufficient points,4=flux,8=temperature ceiling. No residual or temperature ceiling by default. Nonpositive flux warning retained even with flux filter disabled.',
                            interpretation='Low/high fitted temperatures and densities are not combined to reconstruct full fit. Model/partial-moment/calibration differences remain. New raw generator implementation not independently verified.',
                            summary=row, output_sha256={name: core.sha256(target / name) for name in OUTPUTS})
                (target / PROVENANCE).write_text(json.dumps(meta, indent=2, default=core.json_default) + '\n')
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            rows.append(row)
            save_summary()
            print(json.dumps(row), flush=True)
    return int(any(row['status'] == 'failed' for row in rows))


def run_month(args, dates):
    inventory = []
    for date in dates:
        row = dict(date=date, status='ready')
        try:
            events = core.discover_events(args.dataset, date)
            row['n_events'] = len(events)
            core.check_events(events, date)
            row['inputs'] = {k: str(v) for k, v in input_files(args, date).items()}
        except (OSError, ValueError) as exc:
            row.update(status='missing_or_unsupported_input', error=str(exc))
        inventory.append(row)
        print(json.dumps(row), flush=True)
    print(f"{args.month}: {len(dates)} days, {sum(r.get('n_events', 0) for r in inventory)} events, "
          f"{sum(r['status'] != 'ready' for r in inventory)} days with missing/unsupported inputs", flush=True)
    if args.dry_run:
        return int(any(r['status'] != 'ready' for r in inventory))
    base = args.output_root / args.dataset.name if args.output_root else args.dataset
    summary_dir = base / core.OUTPUT_DIR / VERSION
    logs = summary_dir / 'logs' / args.month
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for info in inventory:
        date = info['date']
        log_path = logs / (date + '.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--date', date,
                   '--dataset', str(args.dataset)]
        for flag in ('raw_root', 'fit_1min_root', 'fit_raw_root', 'scratch', 'output_root', *options(args)):
            value = getattr(args, flag)
            if value is not None:
                command += ['--' + flag.replace('_', '-'), str(value)]
        for flag in ('force', 'download_missing'):
            if getattr(args, flag):
                command.append('--' + flag.replace('_', '-'))
        print(f'Processing {date}; log: {log_path}', flush=True)
        with log_path.open('w') as log:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as proc:
                try:
                    returncode = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise
        rows.append(dict(date=date, n_events=info.get('n_events', 0), status='complete' if returncode == 0 else 'failed',
                         returncode=returncode, log=str(log_path)))
        pd.DataFrame(rows).to_csv(summary_dir / f'month_{args.month.replace("-", "")}_summary.csv', index=False)
        print(json.dumps(rows[-1]), flush=True)
    return int(any(row['returncode'] != 0 for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument('--date', help='UTC YYYY-MM-DD')
    period.add_argument('--month', help='UTC YYYYMM or YYYY-MM')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--raw-root', type=Path, default=core.RAW_ROOT)
    parser.add_argument('--fit-1min-root', type=Path, default=FIT_BASE / 'fitmom-1min/1min')
    parser.add_argument('--fit-raw-root', type=Path, default=FIT_BASE / 'fitmom-raw')
    parser.add_argument('--scratch', type=Path, default=Path('/tmp'))
    parser.add_argument('--low-fit-flux-min', type=float, default=1e4)
    parser.add_argument('--high-fit-flux-min', type=float, default=None)
    parser.add_argument('--raw-fit-flux-min', type=float, default=None)
    parser.add_argument('--fit-temperature-max', type=float, default=None, help='Optional QUALITY cut [eV]')
    parser.add_argument('--plot-temperature-max', type=float, default=None, help='Optional DISPLAY cap [eV]; off-scale points marked, raw values retained')
    parser.add_argument('--raw-match-tolerance-ms', type=float, default=1.)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--download-missing', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.date:
            if pd.Timestamp(args.date).strftime('%Y-%m-%d') != args.date:
                raise ValueError('Use YYYY-MM-DD')
        else:
            if len(args.month) == 6 and args.month.isdigit():
                args.month = args.month[:4] + '-' + args.month[4:]
            if pd.Timestamp(args.month + '-01').strftime('%Y-%m') != args.month:
                raise ValueError('Use YYYYMM or YYYY-MM')
        for name, value in options(args).items():
            if value is not None and (not np.isfinite(value) or value < 0 or (not name.endswith('flux_min') and value == 0)):
                raise ValueError(f'Invalid positive/nonnegative option {name}: {value}')
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dataset is None:
        candidates = [p for p in (core.ANALYSIS_ROOT / 'auto').iterdir()
                      if p.is_dir() and core.selected_dates(p, args.date, args.month)]
        if len(candidates) != 1:
            parser.error(f'Expected one dataset; found {len(candidates)}; specify --dataset')
        args.dataset = candidates[0]
    dates = core.selected_dates(args.dataset, args.date, args.month)
    if not dates:
        parser.error('No candidate folders in requested period')
    return run_month(args, dates) if args.month else run_day(args)


if __name__ == '__main__':
    raise SystemExit(main())
