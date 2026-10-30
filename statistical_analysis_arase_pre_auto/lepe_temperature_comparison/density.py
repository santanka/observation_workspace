"""Independent LEP-e partial/model density comparison against native HFA L3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import compare as core
import multiband as mb
import numpy as np
import pandas as pd

VERSION = 'density_hfa_v1'
OUTPUTS = ('density_comparison.png', 'density_comparison.pdf', 'native_comparison.csv',
           'minute_comparison.csv', 'diagnostics.npz')
PROVENANCE = 'provenance.json'


def options(args):
    return {name: getattr(args, name) for name in (
        'low_fit_flux_min', 'high_fit_flux_min', 'raw_fit_flux_min',
        'raw_match_tolerance_ms', 'hfa_match_tolerance_s', 'plot_density_max')}


def code_hashes():
    return {**mb.code_hashes(), Path(__file__).name: core.sha256(__file__)}


def input_files(args, date):
    day = pd.Timestamp(date)
    tag = f'{day:%Y%m%d}'
    patterns = {
        'lepe': (args.raw_root / 'lepe/l2/3dflux' / f'{day:%Y/%m}', f'*_{tag}_v05_*.cdf'),
        'hfa': (args.raw_root / 'pwe/hfa/l3' / f'{day:%Y/%m}', f'erg_pwe_hfa_l3_1min_{tag}_v*.cdf'),
    }
    for band, config in mb.BANDS.items():
        root = args.fit_raw_root if band == 'all' else args.fit_1min_root
        name = (f'erg-lepe-fitmom-{config["resolution"]}-{config["fit_min"]:.0f}eV-'
                f'{config["fit_max"]:.0f}eV-{tag}-000000-{tag}-240000-v2.nc')
        patterns[band] = (root / f'{day:%Y}', name)
    paths, missing = {}, []
    for key, (root, pattern) in patterns.items():
        files = sorted(root.glob(pattern))
        if files:
            paths[key] = files[-1]
        else:
            missing.append(str(root / pattern))
    if missing:
        raise FileNotFoundError('Required local inputs missing: ' + '; '.join(missing))
    return paths


def pi_density(frame, flux_min=None):
    out = frame.copy()
    n = frame['ne'].to_numpy(dtype=float)
    reason = np.zeros(len(frame), dtype=np.uint8)
    reason[~(np.isfinite(n) & (n > 0))] |= 1
    temperatures = frame[['T_para', 'T_perp']].to_numpy(dtype=float)
    reason[~(np.isfinite(temperatures) & (temperatures > 0)).all(axis=1)] |= 2
    count = frame.num_valid_data.to_numpy(dtype=float)
    reason[~(np.isfinite(count) & (count > 3))] |= 4
    flux = frame[['eflux_para_mean', 'eflux_perp_mean']].to_numpy(dtype=float)
    out['flux_warning'] = ~(np.isfinite(flux) & (flux > 0)).all(axis=1)
    if flux_min is not None:
        reason[~(np.isfinite(flux) & (flux > flux_min)).all(axis=1)] |= 8
    out['density_raw_cm3'] = n * 1e-6
    out['density_cm3'] = np.where(reason == 0, n * 1e-6, np.nan)
    out['valid'] = reason == 0
    out['reason'] = reason
    return out


def integral_quality(frame, packet_mask):
    """Density mask deliberately independent of temperature and magnetic validity."""
    out = frame.copy()
    n = out.density_raw_cm3.to_numpy(dtype=float)
    reason = np.zeros(len(out), dtype=np.uint8)
    reason[~(np.isfinite(n) & (n > 0))] |= 1
    reason[(out.packet_info_raw.to_numpy(dtype='int64') & packet_mask) != 0] |= 2
    reason[out.n_finite_flux.to_numpy() == 0] |= 4
    out['density_cm3'] = np.where(reason == 0, n, np.nan)
    out['valid'] = reason == 0
    out['reason'] = reason
    return out


def compute_density(cdf, source, start, end, band, packet_mask, auxiliary_field=(0., 0., 1.)):
    """Public moment engine; a nonzero auxiliary B only serves discarded tensor diagnostics."""
    import pyspedas as psp
    from cdflib import cdfepoch
    times = np.asarray(cdfepoch.to_datetime(cdf.varget('epoch'))).astype('datetime64[ns]')
    if np.isnat(times).any() or np.any(np.diff(times).astype('int64') <= 0):
        raise ValueError('Invalid/duplicate LEP-e epochs')
    first, stop = np.searchsorted(times, [start.to_datetime64(), end.to_datetime64()])
    if first == stop:
        raise ValueError('No LEP-e records in requested interval')
    time = times[first:stop]
    unix = time.astype('int64') / 1e9
    flux = core.read_var(cdf, 'fedu', first, stop - 1).transpose(0, 2, 3, 1)
    energy = core.read_var(cdf, 'fedu_energy', first, stop - 1)
    packet = core.read_var(cdf, 'packet_info', first, stop - 1)
    config = mb.BANDS[band]
    channel = np.isfinite(energy) & (energy >= config['lower']) & (energy <= config['upper'])
    if not channel.any():
        raise ValueError('No energy channels in requested band')
    selected_flux = np.broadcast_to(channel[:, :, None, None], flux.shape)
    finite = selected_flux & np.isfinite(flux)
    psp.del_data('*')
    psp.store_data(core.PREFIX, data={'x': unix, 'y': flux, 'v1': energy,
                                     'v2': np.arange(flux.shape[2]), 'v3': np.arange(16)},
                   attr_dict={'CDF': {'FILENAME': str(source), 'VATT': cdf.varattsget('fedu'),
                                      'GATT': cdf.globalattsget()}})
    field = np.asarray(auxiliary_field, dtype=float)
    if field.shape != (3,) or not np.isfinite(field).all() or np.linalg.norm(field) == 0:
        raise ValueError('Auxiliary field must be a finite nonzero vector')
    psp.store_data('density_auxiliary_field', data={'x': unix, 'y': np.tile(field, (len(time), 1))})
    names = psp.projects.erg.erg_lep_part_products(
        core.PREFIX, outputs=['moments'], energy=[config['lower'], config['upper']],
        mag_name='density_auxiliary_field')
    if not names or core.PREFIX + '_density' not in names:
        raise RuntimeError('Public moment engine returned no density')
    result = psp.get_data(core.PREFIX + '_density')
    np.testing.assert_array_equal(result.times, unix)
    frame = pd.DataFrame(dict(density_raw_cm3=result.y, packet_info_raw=packet,
                              n_finite_flux=finite.sum(axis=(1, 2, 3)),
                              n_selected_flux=selected_flux.sum(axis=(1, 2, 3)),
                              n_negative_flux=(finite & (flux < 0)).sum(axis=(1, 2, 3))),
                         index=pd.DatetimeIndex(time, name='time_utc'))
    frame = integral_quality(frame, packet_mask)
    centers = np.unique(energy[channel])
    return frame, dict(actual_energy_centers_eV=centers,
                       integral_center_range_eV=[config['lower'], None if np.isinf(config['upper']) else config['upper']])


def hfa_quality(frame):
    out = frame.copy()
    # Preserve raw float32 separately; use float64 for reproducible CSV ratios.
    n, flag = frame.density_raw_cm3.to_numpy(dtype=float), frame.quality_raw.to_numpy()
    reason = np.zeros(len(out), dtype=np.uint8)
    reason[~(np.isfinite(n) & (n > 0))] |= 1
    reason[~(np.isfinite(flag) & (flag == 0))] |= 2
    out['density_cm3'] = np.where(reason == 0, n, np.nan)
    out['valid'] = reason == 0
    out['reason'] = reason
    return out


def load_hfa(path):
    from cdflib import CDF, cdfepoch
    cdf = CDF(path)
    attrs = {name: cdf.varattsget(name) for name in ('Epoch', 'ne_mgf', 'quality_flag')}
    if str(attrs['ne_mgf'].get('UNITS')).lower() not in ('/cc', '1/cm^3', 'cm^-3'):
        raise ValueError('Unexpected HFA density units')
    t = np.asarray(cdfepoch.to_datetime(cdf.varget('Epoch'))).astype('datetime64[ns]')
    if not len(t) or np.isnat(t).any() or np.any(np.diff(t).astype('int64') <= 0):
        raise ValueError('Invalid/duplicate HFA epochs')
    frame = pd.DataFrame(dict(density_raw_cm3=core.read_var(cdf, 'ne_mgf'),
                              quality_raw=core.read_var(cdf, 'quality_flag'),
                              fuhr_kHz=core.read_var(cdf, 'Fuhr')),
                         index=pd.DatetimeIndex(t, name='time_utc'))
    phase = (t - t.astype('datetime64[m]')).astype('int64') / 1e9
    return hfa_quality(frame), dict(attributes=attrs, data_version=cdf.globalattsget().get('DATA_VERSION'),
                                    seconds_within_minute_min=float(phase.min()),
                                    seconds_within_minute_max=float(phase.max()),
                                    timing_note='Epoch CATDESC says beginning; observed labels near minute centers. No epoch shift applied; nearest matching to PI minute centers only.')


def align_hfa(minute_index, hfa, tolerance_s=1.):
    centers = minute_index + pd.Timedelta(seconds=30)
    aligned = mb.match_raw(centers, hfa, tolerance_ms=tolerance_s * 1000)
    aligned.index = minute_index
    return aligned.rename(columns={name: ('hfa_' + name[4:] if name.startswith('fit_') else
                                          'hfa_time_utc' if name == 'pi_time_utc' else
                                          'hfa_source_index' if name == 'pi_source_index' else
                                          'hfa_' + name) for name in aligned})


def minute_table(natives, fits, hfa, lo, hi, tolerance_s):
    index = pd.date_range(lo, hi, freq='1min', inclusive='left', name='time_utc')
    table = align_hfa(index, hfa, tolerance_s)
    for band in mb.BANDS:
        n, f = natives[band], fits[band]
        table[band + '__integral_density_cm3'] = n.density_cm3.resample('1min').mean().reindex(index)
        table[band + '__integral_n_total'] = n.valid.resample('1min').size().reindex(index, fill_value=0)
        table[band + '__integral_n_valid'] = n.valid.resample('1min').sum().reindex(index, fill_value=0)
        table[band + '__integral_valid_fraction'] = (table[band + '__integral_n_valid'] /
                                                    table[band + '__integral_n_total'].replace(0, np.nan))
        if band == 'all':
            table[band + '__fit_density_cm3'] = f.density_cm3.resample('1min').mean().reindex(index)
            table[band + '__fit_density_raw_cm3'] = f.density_raw_cm3.resample('1min').mean().reindex(index)
            table[band + '__fit_n_total'] = f.valid.resample('1min').size().reindex(index, fill_value=0)
            table[band + '__fit_n_valid'] = f.valid.resample('1min').sum().reindex(index, fill_value=0)
        else:
            minute_fit = f.reindex(index)
            for name in minute_fit:
                table[band + '__fit_' + name] = minute_fit[name]
            table[band + '__fit_n_total'] = index.isin(f.index).astype(int)
            table[band + '__fit_n_valid'] = minute_fit.valid.fillna(False).astype(int)
        for key in ('integral', 'fit'):
            table[f'{band}__{key}_over_hfa'] = table[f'{band}__{key}_density_cm3'] / table.hfa_density_cm3
    return table


def ratio_stats(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    stats = dict(n_paired_minutes=len(values))
    for quantile, name in [(.25, 'q25'), (.5, 'median'), (.75, 'q75')]:
        stats[name] = float(np.quantile(values, quantile)) if len(values) else np.nan
    return {prefix + '_' + name: value for name, value in stats.items()}


def plot_event(natives, fits, hfa, table, start, end, target, cap=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, layout='constrained')
    lo, hi = table.index[0], table.index[-1] + pd.Timedelta(minutes=1)
    center = table.index + pd.Timedelta(seconds=30)
    titles = {'all': 'Full: Integral E >= 67 eV / PI raw 50-20000 eV',
              'low': 'Low: Integral 67-1000 eV / PI fit 50-1000 eV',
              'high': 'High: Integral / PI fit 1000-20000 eV'}
    for col, band in enumerate(mb.BANDS):
        ax, ratio_ax = axes[:, col]
        n, f = natives[band], fits[band]
        above = []
        above.append(mb.draw_series(ax, n.index, n.density_cm3, cap=cap,
                                    color='C0', linestyle='None', alpha=.25, ms=2))
        if band == 'all':
            above.append(mb.draw_series(ax, f.index, f.density_cm3, cap=cap,
                                        color='C1', linestyle='None', alpha=.25, ms=2))
        above.append(mb.draw_series(ax, center, table[band + '__integral_density_cm3'], cap=cap,
                                    color='C0', label='Integral partial n: 1-min mean',
                                    marker='o', linestyle='-', ms=3, lw=1))
        above.append(mb.draw_series(ax, center, table[band + '__fit_density_cm3'], cap=cap, color='C1',
                                    label='PI model n: mean of raw n' if band == 'all' else 'PI model n: fit to mean PSD',
                                    marker='s', minute=band != 'all', ms=3, lw=1))
        rejected = ~f.valid
        if rejected.any():
            times = f.index if band == 'all' else f.index + pd.Timedelta(seconds=30)
            above.append(mb.draw_series(ax, times[rejected], f.loc[rejected, 'density_raw_cm3'],
                                        cap=cap, color='.5', marker='x', linestyle='None', ms=3,
                                        label='PI rejected'))
        above.append(mb.draw_series(ax, hfa.index, hfa.density_cm3, cap=cap, color='k', marker='D',
                                    linestyle='None', ms=4, label='HFA reference: native 1-min epoch'))
        if (~hfa.valid).any():
            above.append(mb.draw_series(ax, hfa.index[~hfa.valid], hfa.loc[~hfa.valid, 'density_raw_cm3'],
                                        cap=cap, color='.6', marker='x', linestyle='None', ms=4,
                                        label='HFA rejected'))
        ax.set(yscale='log', ylabel=r'$n_e$ [cm$^{-3}$]', title=titles[band])
        if cap is not None and ax.get_ylim()[1] > cap * 1.15:
            ax.set_ylim(top=cap * 1.15)
        off = [(count, maximum) for count, maximum in above if count]
        if off:
            ax.text(.02, .98, f'Above display limit: {sum(c for c, _ in off)}; max={max(v for _, v in off):.2g}',
                    transform=ax.transAxes, va='top', fontsize=7)
        ax.legend(fontsize=7, loc='best')
        for key, color, marker in [('integral', 'C0', 'o'), ('fit', 'C1', 's')]:
            ratio_ax.plot(center, table[f'{band}__{key}_over_hfa'], marker=marker, color=color,
                          ms=3, lw=1, label=key + ' / HFA')
        ratio_ax.axhline(1., color='k', ls='--', lw=.8)
        ratio_ax.set(yscale='log', ylabel='Density ratio to HFA', xlabel='UTC')
        ratio_ax.legend(fontsize=8)
        for item in (ax, ratio_ax):
            item.axvspan(lo, start, color='.8', alpha=.3)
            item.axvspan(end, hi, color='.8', alpha=.3)
            item.set_xlim(lo, hi)
            item.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            item.grid(which='both', alpha=.2)
    fig.suptitle(f'LEP-e density | {start:%Y-%m-%d %H:%M:%S} - {end:%H:%M:%S} UTC\n'
                 'Integral = measured partial density; PI = model normalization; HFA = independent reference', fontsize=12)
    for ext in ('png', 'pdf'):
        fig.savefig(target / ('density_comparison.' + ext), dpi=160)
    plt.close(fig)


def result_signature(inputs, marker, args, start, end):
    payload = dict(version=VERSION, inputs=inputs, code=code_hashes(), options=options(args),
                   range_state_sha256=core.sha256(marker), start=str(start), end=str(end),
                   packages={name: core.version(name) for name in mb.PACKAGES})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def reusable(target, signature):
    try:
        meta = json.loads((target / PROVENANCE).read_text())
        if meta['signature'] != signature or set(meta['output_sha256']) != set(OUTPUTS):
            return None
        if any(core.sha256(target / name) != digest for name, digest in meta['output_sha256'].items()):
            return None
        return meta['summary']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def output_day(args, date):
    root = args.output_root / args.dataset.name if args.output_root else args.dataset
    return root / date


def run_day(args):
    events = core.discover_events(args.dataset, args.date)
    core.check_events(events, args.date)
    paths = input_files(args, args.date)
    print(f'{args.date}: {len(events)} events; ' + ', '.join(f'{k}={v}' for k, v in paths.items()), flush=True)
    if args.dry_run:
        return 0
    inputs = {key: dict(path=str(path), sha256=core.sha256(path)) for key, path in paths.items()}
    rows, pending = [], []
    summary_dir = output_day(args, args.date) / core.OUTPUT_DIR / VERSION
    summary_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        folder, start, end, _, marker = event
        target = output_day(args, args.date) / folder.name / core.OUTPUT_DIR / VERSION
        stamp = result_signature(inputs, marker, args, start, end)
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
    fits, fit_meta = {}, {}
    for band in mb.BANDS:
        f, fit_meta[band] = mb.load_fit(paths[band], band)
        threshold = getattr(args, {'all': 'raw_fit_flux_min', 'low': 'low_fit_flux_min', 'high': 'high_fit_flux_min'}[band])
        fits[band] = pi_density(f, threshold)
    hfa_day, hfa_meta = load_hfa(paths['hfa'])
    from cdflib import CDF
    with core.staged_cdf(paths['lepe'], args.scratch) as lp:
        lepe = CDF(lp)
        for (folder, start, end, state, marker), target, stamp in pending:
            row = dict(event=folder.name, start_utc=str(start), end_utc=str(end), status='failed')
            try:
                target.mkdir(parents=True, exist_ok=True)
                (target / PROVENANCE).unlink(missing_ok=True)
                lo, hi = start.floor('min'), end.ceil('min')
                natives, selected_fits, integral_meta = {}, {}, {}
                print(f'Computing density {folder.name}', flush=True)
                for band in mb.BANDS:
                    natives[band], integral_meta[band] = compute_density(
                        lepe, lp, lo, hi, band, int(fit_meta[band]['attributes']['packet_info_bitmask']))
                    pad = pd.Timedelta(milliseconds=args.raw_match_tolerance_ms) if band == 'all' else pd.Timedelta(0)
                    f = fits[band]
                    selected_fits[band] = f[(f.index >= lo - pad) & (f.index < hi + pad)].copy()
                hfa = hfa_day[(hfa_day.index >= lo) & (hfa_day.index < hi)].copy()
                table = minute_table(natives, selected_fits, hfa_day, lo, hi, args.hfa_match_tolerance_s)
                raw_match = mb.match_raw(natives['all'].index, selected_fits['all'], args.raw_match_tolerance_ms)
                raw_match['integral_over_fit'] = natives['all'].density_cm3 / raw_match.fit_density_cm3
                in_event = (raw_match.index >= start) & (raw_match.index < end)
                row['raw_n_matched_in_event'] = int(((raw_match.status == 'matched') & in_event).sum())
                row['raw_n_unmatched_in_event'] = int(((raw_match.status != 'matched') & in_event).sum())
                row['hfa_n_time_matched_minutes'] = int((table.hfa_status == 'matched').sum())
                row['hfa_n_valid_minutes'] = int(np.isfinite(table.hfa_density_cm3).sum())
                for band in mb.BANDS:
                    row[band + '_n_valid_integral_minutes'] = int(np.isfinite(table[band + '__integral_density_cm3']).sum())
                    row[band + '_n_valid_fit_minutes'] = int(np.isfinite(table[band + '__fit_density_cm3']).sum())
                    for key in ('integral', 'fit'):
                        row.update(ratio_stats(table[f'{band}__{key}_over_hfa'], band + '_' + key + '_over_hfa'))
                native_csv = pd.concat([frame.add_prefix(band + '__') for band, frame in natives.items()], axis=1)
                native_csv = native_csv.join(raw_match.add_prefix('raw_match__'), how='outer')
                native_csv = native_csv.join(selected_fits['all'].add_prefix('pi_raw__'), how='outer')
                native_csv = native_csv.join(hfa.add_prefix('hfa__'), how='outer').sort_index()
                native_csv['record_has_integral'] = native_csv.index.isin(natives['all'].index)
                native_csv['record_has_pi_raw'] = native_csv.index.isin(selected_fits['all'].index)
                native_csv['record_has_hfa'] = native_csv.index.isin(hfa.index)
                native_csv['within_event'] = (native_csv.index >= start) & (native_csv.index < end)
                diagnostic = {}
                for key, frame in {**natives, **{'pi_' + k: v for k, v in selected_fits.items()}, 'hfa': hfa}.items():
                    diagnostic[key + '__time_utc'] = frame.index.to_numpy(dtype='datetime64[ns]')
                    for name in frame:
                        diagnostic[key + '__' + name] = frame[name].to_numpy()
                plot_event(natives, selected_fits, hfa, table, start, end, target, args.plot_density_max)
                native_csv.to_csv(target / 'native_comparison.csv', index_label='time_utc')
                table.to_csv(target / 'minute_comparison.csv', index_label='time_utc')
                np.savez_compressed(target / 'diagnostics.npz', **diagnostic)
                row['status'] = 'complete'
                meta = dict(version=VERSION, signature=stamp, created_utc=datetime.now(timezone.utc).isoformat(),
                            event_start_utc=str(start), event_end_utc=str(end), original_event_folder=str(folder),
                            range_id=state['range_id'], batch_status=state['status'], inputs=inputs,
                            range_state_path=str(marker), range_state_sha256=core.sha256(marker),
                            code_sha256=code_hashes(), packages={name: core.version(name) for name in mb.PACKAGES},
                            options=options(args), integral_profiles=integral_meta, pi_profiles=fit_meta, hfa=hfa_meta,
                            units='All displayed densities cm^-3. PI original ne m^-3 multiplied by 1e-6; PI original ne retained. HFA /cc and Integral cm^-3 unchanged.',
                            integral_quality='Finite positive density; PI packet_info bitmask; >=1 finite flux cell in band. No temperature/magnetic validity mask. Negative flux retained by public routine.',
                            auxiliary_field='Constant nonzero vector only for discarded tensor diagnostics in the public all-moment API. Scalar density does not depend on this direction. No measured MGF required.',
                            pi_quality='Finite positive ne, both temperatures finite positive, num_valid_data>3, configured BOTH directional flux thresholds. No density/temperature/residual upper cuts. Bits 1=density,2=temperatures,4=points,8=flux.',
                            hfa_quality='Finite positive ne_mgf and quality_flag exactly zero. Raw bit field retained. Bits 1=density,2=flag.',
                            timing='UTC. Native raw matching within configured tolerance; one-to-one, no interpolation. HFA matched one-to-one to nominal minute centers within configured tolerance; original Epoch retained, no shift or interval-support assumption. HFA invalid/missing minutes stay missing. Complete minute bins overlapping event used for minute statistics; native summary restricted to [start,end).',
                            averaging='Arithmetic means of valid native densities, not exposure-weighted. Full PI curve is mean of native fitted densities, not fit to mean distribution. Low/high PI are provided fits to minute-averaged distributions.',
                            interpretation='Integral is partial measured density; PI ne is full model normalization/extrapolation; HFA is independent reference, not fixed truth. Low/high PI ne are not added. Ratio<1 is not by itself an integration error. Same-band forward-model density is not computed in this version.',
                            summary=row, output_sha256={name: core.sha256(target / name) for name in OUTPUTS})
                (target / PROVENANCE).write_text(json.dumps(meta, indent=2, default=core.json_default) + '\n')
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            rows.append(row)
            save_summary()
            print(json.dumps(row), flush=True)
    return int(any(row['status'] == 'failed' for row in rows))


def pooled_minute_summary(tables):
    pooled = pd.concat(tables).sort_index()
    # Adjacent candidate events may share a complete minute. Count it only once.
    for _, group in pooled[pooled.index.duplicated(keep=False)].groupby(level=0):
        columns = [name for name in pooled if name.endswith('_over_hfa')]
        values = group[columns].to_numpy(dtype=float)
        np.testing.assert_allclose(values, np.broadcast_to(values[0], values.shape), rtol=1e-10, equal_nan=True)
    pooled = pooled[~pooled.index.duplicated(keep='first')]
    rows = []
    for band in mb.BANDS:
        row = dict(band=band, n_unique_minutes=len(pooled), n_hfa_valid_minutes=int(np.isfinite(pooled.hfa_density_cm3).sum()))
        for key in ('integral', 'fit'):
            row.update(ratio_stats(pooled[f'{band}__{key}_over_hfa'], key + '_over_hfa'))
        rows.append(row)
    return pd.DataFrame(rows)


def run_month(args, dates):
    inventory = []
    for date in dates:
        row = dict(date=date, status='ready')
        try:
            events = core.discover_events(args.dataset, date)
            row['n_events'] = len(events)
            core.check_events(events, date)
            row['inputs'] = {key: str(path) for key, path in input_files(args, date).items()}
        except (OSError, ValueError) as exc:
            row.update(status='missing_or_unsupported_input', error=str(exc))
        inventory.append(row)
        print(json.dumps(row), flush=True)
    print(f"{args.month}: {len(dates)} days, {sum(r.get('n_events', 0) for r in inventory)} events, "
          f"{sum(r['status'] != 'ready' for r in inventory)} days with missing inputs", flush=True)
    if args.dry_run:
        return int(any(r['status'] != 'ready' for r in inventory))
    base = args.output_root / args.dataset.name if args.output_root else args.dataset
    summary_dir = base / core.OUTPUT_DIR / VERSION
    logs = summary_dir / 'logs' / args.month
    logs.mkdir(parents=True, exist_ok=True)
    rows, event_rows, minute_tables = [], [], []
    for info in inventory:
        date = info['date']
        log_path = logs / (date + '.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--date', date, '--dataset', str(args.dataset)]
        for flag in ('raw_root', 'fit_1min_root', 'fit_raw_root', 'scratch', 'output_root', *options(args)):
            value = getattr(args, flag)
            if value is not None:
                command += ['--' + flag.replace('_', '-'), str(value)]
        if args.force:
            command.append('--force')
        print(f'Processing {date}; log: {log_path}', flush=True)
        with log_path.open('w') as log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as proc:
            try:
                code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
        rows.append(dict(date=date, status='complete' if code == 0 else 'failed', returncode=code, log=str(log_path)))
        if code == 0:
            day = output_day(args, date)
            events = pd.read_csv(day / core.OUTPUT_DIR / VERSION / 'event_summary.csv')
            event_rows.append(events)
            for event in events.event:
                table = pd.read_csv(day / event / core.OUTPUT_DIR / VERSION / 'minute_comparison.csv')
                table.index = pd.DatetimeIndex(pd.to_datetime(table.pop('time_utc')), name='time_utc')
                minute_tables.append(table)
        pd.DataFrame(rows).to_csv(summary_dir / f'month_{args.month.replace("-", "")}_days.csv', index=False)
        if event_rows:
            pd.concat(event_rows).to_csv(summary_dir / f'month_{args.month.replace("-", "")}_events.csv', index=False)
            pooled_minute_summary(minute_tables).to_csv(summary_dir / f'month_{args.month.replace("-", "")}_ratios.csv', index=False)
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
    parser.add_argument('--fit-1min-root', type=Path, default=mb.FIT_BASE / 'fitmom-1min/1min')
    parser.add_argument('--fit-raw-root', type=Path, default=mb.FIT_BASE / 'fitmom-raw')
    parser.add_argument('--scratch', type=Path, default=Path('/tmp'))
    parser.add_argument('--low-fit-flux-min', type=float, default=1e4)
    parser.add_argument('--high-fit-flux-min', type=float, default=None)
    parser.add_argument('--raw-fit-flux-min', type=float, default=None)
    parser.add_argument('--raw-match-tolerance-ms', type=float, default=1.)
    parser.add_argument('--hfa-match-tolerance-s', type=float, default=1.)
    parser.add_argument('--plot-density-max', type=float, default=None, help='Display cap only, cm^-3; original values retained')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
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
                raise ValueError(f'Invalid option: {name}')
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.dataset is None:
        datasets = [p for p in (core.ANALYSIS_ROOT / 'auto').iterdir()
                    if p.is_dir() and core.selected_dates(p, args.date, args.month)]
        if len(datasets) != 1:
            parser.error(f'Expected one dataset; found {len(datasets)}; use --dataset')
        args.dataset = datasets[0]
    dates = core.selected_dates(args.dataset, args.date, args.month)
    if not dates:
        parser.error('No candidate folders in requested period')
    return run_month(args, dates) if args.month else run_day(args)


if __name__ == '__main__':
    raise SystemExit(main())
