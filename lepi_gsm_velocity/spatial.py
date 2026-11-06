"""Offline sample-mean GSM velocity maps on the KAW L(90deg)-MLAT mesh."""
import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lepi_spatial_matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
from core import interpolate, series

SPECIES = {'FPDU': 'H+', 'FHEDU': 'He+', 'FODU': 'O+'}
L_EDGES = np.arange(3., 12., 1.)
LAT_EDGES = np.arange(-50., 55., 5.)
MLT_EDGES = np.arange(0., 25., 1.)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def parse(text):
    date = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    return date.replace(tzinfo=dt.timezone.utc) if date.tzinfo is None else date.astimezone(dt.timezone.utc)


def read_orbit(path):
    import cdflib
    with cdflib.CDF(path) as cdf:
        times = cdflib.cdfepoch.to_datetime(cdf.varget('epoch')).astype('datetime64[ns]').astype(np.int64) / 1e9
        pos = np.asarray(cdf.varget('pos_rmlatmlt'), float)
        ls = np.asarray(cdf.varget('pos_Lm'), float)
        for name, values in (('pos_rmlatmlt', pos), ('pos_Lm', ls)):
            attrs = cdf.varattsget(name)
            for key, op in (('FILLVAL', np.equal), ('VALIDMIN', np.less), ('VALIDMAX', np.greater)):
                if key in attrs:
                    bound = np.asarray(attrs[key], dtype=float)
                    values[op(values, bound)] = np.nan
    if pos.shape != (len(times), 3) or ls.shape != pos.shape:
        raise ValueError('Unexpected orbit array shape: ' + str(path))
    # Do not remove invalid slots or unwrap angles across gaps.
    values = np.column_stack((ls[:, 0], pos[:, 1], pos[:, 2]))
    valid = (np.isfinite(values).all(axis=1) & (values[:, 0] > 0)
             & (values[:, 0] < 1e6) & (abs(values[:, 1]) <= 90)
             & (values[:, 2] >= 0) & (values[:, 2] <= 24))
    values[~valid] = np.nan
    return series(times, values)


def orbit_at(times, support_times, values, max_gap=120.):
    # Local shortest-arc interpolation, so 23.9 -> 0.1 passes through midnight.
    coords, valid, brackets = interpolate(support_times, values[:, :2], times, max_gap)
    left, right = brackets['left'], brackets['right']
    ml, mr = values[left, 2], values[right, 2]
    gap = brackets['gap_seconds']
    fraction = np.divide(times - brackets['left_time_unix'], gap,
                         out=np.zeros(len(times)), where=gap > 0)
    mlt = np.mod(ml + fraction * ((mr - ml + 12.) % 24. - 12.), 24.)
    valid &= np.isfinite(ml) & np.isfinite(mr)
    coords[~valid] = np.nan
    mlt[~valid] = np.nan
    return coords[:, 0], coords[:, 1], mlt, valid


def sector_mask(mlt, sector):
    return ((mlt >= 18) | (mlt < 6)) if sector == 'night' else ((mlt >= 6) & (mlt < 18))


class Accumulator:
    def __init__(self, l_edges=L_EDGES, lat_edges=LAT_EDGES):
        self.l_edges, self.lat_edges = l_edges, lat_edges
        shape = (len(l_edges) - 1, len(lat_edges) - 1)
        self.n = np.zeros(shape, dtype=np.int64)
        self.total = np.zeros(shape + (3,), dtype=np.float64)
        self.absolute = np.zeros_like(self.total)
        self.days = np.zeros(shape, dtype=np.int64)
        self.last_day = np.full(shape, '', dtype='U10')

    def add(self, l_value, latitude, velocity, valid, day):
        good = valid & np.isfinite(l_value) & np.isfinite(latitude) & np.isfinite(velocity).all(axis=1)
        x, y, v = l_value[good], latitude[good], np.asarray(velocity[good], dtype=np.float64)
        ix = np.searchsorted(self.l_edges, x, side='right') - 1
        iy = np.searchsorted(self.lat_edges, y, side='right') - 1
        # All bins half-open; the final right edge is included.
        ix[x == self.l_edges[-1]] = self.n.shape[0] - 1
        iy[y == self.lat_edges[-1]] = self.n.shape[1] - 1
        inside = (ix >= 0) & (ix < self.n.shape[0]) & (iy >= 0) & (iy < self.n.shape[1])
        ix, iy, v = ix[inside], iy[inside], v[inside]
        np.add.at(self.n, (ix, iy), 1)
        np.add.at(self.total, (ix, iy), v)
        np.add.at(self.absolute, (ix, iy), np.abs(v))
        cells = np.unique(np.column_stack((ix, iy)), axis=0)
        for i, j in cells:
            if self.last_day[i, j] != day:
                self.days[i, j] += 1
                self.last_day[i, j] = day
        return int(inside.sum())

    def means(self):
        divisor = self.n[..., None]
        return tuple(np.divide(a, divisor, out=np.full_like(a, np.nan), where=divisor > 0)
                     for a in (self.total, self.absolute))


def mesh(l_edges, lat_edges):
    l_grid, lat_grid = np.meshgrid(l_edges, np.deg2rad(lat_edges))
    radius = l_grid * np.cos(lat_grid) ** 2
    return radius * np.cos(lat_grid), radius * np.sin(lat_grid)


def add_earth(ax):
    """Keep white on screen-left even with reversed meridional axes."""
    from matplotlib.patches import Circle, Wedge
    white = (-90, 90) if ax.xaxis_inverted() else (90, 270)
    black = (90, 270) if ax.xaxis_inverted() else (-90, 90)
    ax.add_patch(Wedge((0, 0), 1., *white, facecolor='white', edgecolor='none', zorder=5))
    ax.add_patch(Wedge((0, 0), 1., *black, facecolor='black', edgecolor='none', zorder=5))
    ax.add_patch(Circle((0, 0), 1., facecolor='none', edgecolor='black', lw=1., zorder=6))


def configure_meridional(ax, acc, sector):
    xx, zz = mesh(acc.l_edges, acc.lat_edges)
    for l_value in acc.l_edges:
        gx, gz = mesh(np.array([l_value]), np.linspace(acc.lat_edges[0], acc.lat_edges[-1], 301))
        ax.plot(gx[:, 0], gz[:, 0], color='gray', lw=.4, alpha=.5)
        if l_value in acc.l_edges[::2]:
            ax.text(l_value, -.15, f'{l_value:g}', fontsize=8, ha='center')
    for lat in acc.lat_edges:
        gx, gz = mesh(acc.l_edges, np.array([lat]))
        ax.plot(gx[0], gz[0], color='gray', lw=.4, alpha=.5)
        if lat % 10 == 0 and lat != 0:
            ax.text(gx[0, -1], gz[0, -1], f'{lat:+g}°', fontsize=7)
    ax.set(xlim=(-1.1, float(xx.max()) * 1.07), ylim=(float(zz.min()) * 1.12, float(zz.max()) * 1.12),
           aspect='equal', xlabel=r'Dipole display $L\cos^3(\lambda)$ [$R_E$]',
           ylabel=r'Dipole display $L\cos^2(\lambda)\sin(\lambda)$ [$R_E$]')
    if sector == 'day':
        ax.invert_xaxis()
    add_earth(ax)


def mlt_mesh(l_edges, mlt_edges):
    radius, theta = np.meshgrid(l_edges, 2. * np.pi * mlt_edges / 24.)
    return radius * np.cos(theta), radius * np.sin(theta)


def configure_mlt(ax, acc):
    theta = np.linspace(0., 2. * np.pi, 361)
    outer = acc.l_edges[-1]
    for l_value in acc.l_edges:
        ax.plot(l_value * np.cos(theta), l_value * np.sin(theta), color='gray', lw=.4, alpha=.5)
        if l_value in acc.l_edges[::2]:
            ax.text(l_value / np.sqrt(2), l_value / np.sqrt(2), f'{l_value:g}', fontsize=8)
    for hour in acc.lat_edges[:-1]:
        angle = 2. * np.pi * hour / 24.
        ax.plot(np.array([acc.l_edges[0], outer]) * np.cos(angle),
                np.array([acc.l_edges[0], outer]) * np.sin(angle), color='gray', lw=.4, alpha=.5)
        if hour % 3 == 0:
            ax.text(outer * 1.1 * np.cos(angle), outer * 1.1 * np.sin(angle),
                    f'{hour:02.0f}', ha='center', va='center', fontsize=9)
    ax.set(xlim=(-outer * 1.2, outer * 1.2), ylim=(-outer * 1.2, outer * 1.2), aspect='equal')
    ax.set_axis_off()
    add_earth(ax)


def plot(acc, label, sector, period, output, min_count, min_days, vmax=None, clipped=False,
         coordinate='l_mlat'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    means = acc.means()
    mask = (acc.n < min_count) | (acc.days < min_days)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    xx, zz = (mesh if coordinate == 'l_mlat' else mlt_mesh)(acc.l_edges, acc.lat_edges)
    for row, data in enumerate(means):
        finite = np.abs(data[~mask])
        finite = finite[np.isfinite(finite)]
        limit = vmax[row] if vmax is not None else (float(finite.max()) if finite.size else 1.)
        if limit == 0:
            limit = 1.
        norm = Normalize(-limit if row == 0 else 0., limit)
        for col, component in enumerate('xyz'):
            ax = axes[row, col]
            values = np.ma.array(data[..., col], mask=mask | ~np.isfinite(data[..., col]))
            artist = ax.pcolormesh(xx, zz, values.T, cmap='RdBu_r' if row == 0 else 'viridis', norm=norm)
            if coordinate == 'l_mlat':
                configure_meridional(ax, acc, sector)
            else:
                configure_mlt(ax, acc)
            quantity = rf'v_{{{component}}}' if row == 0 else rf'|v_{{{component}}}|'
            ax.set_title(rf'$\langle {quantity} \rangle$')
            fig.colorbar(artist, ax=ax, label='GSM velocity [km/s]', shrink=.75,
                         extend=('both' if row == 0 else 'max') if clipped else 'neither')
    if coordinate == 'l_mlat':
        hours = '18–06' if sector == 'night' else '06–18'
        selection = f'{sector} ({hours} MLT); L–MLAT'
    else:
        selection = 'L–MLT (00 right, 06 top, 12 left, 18 bottom); −50° ≤ MLAT ≤ 50°'
    fig.suptitle(f'{label}; {period}; {selection}\n'
                 f'Spacecraft-relative partial moments; sample means; N >= {min_count}, days >= {min_days}')
    for suffix in ('.png', '.pdf'):
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)


def save_table(acc, path, coordinate='mlat'):
    signed, absolute = acc.means()
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['l_left', 'l_right', *(['mlat_left_deg', 'mlat_right_deg'] if coordinate == 'mlat' else ['mlt_left_hour', 'mlt_right_hour']), 'n_records', 'n_days'] +
                        [f'{kind}_{c}_kms' for kind in ('mean', 'mean_abs') for c in 'xyz'])
        for i, j in np.ndindex(acc.n.shape):
            writer.writerow([acc.l_edges[i], acc.l_edges[i+1], acc.lat_edges[j], acc.lat_edges[j+1],
                             acc.n[i, j], acc.days[i, j], *signed[i, j], *absolute[i, j]])


def orbit_path(root, day):
    files = []
    for base in (root / 'ergsc', root):
        files.extend((base / f'satellite/erg/orb/def/{day:%Y}').glob(f'erg_orb_l2_{day:%Y%m%d}_v??.cdf'))
    if not files:
        raise FileNotFoundError(f'Local definitive ORB missing: {day:%Y-%m-%d}')
    return max(files, key=lambda p: (p.name, 'ergsc' in p.parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gsm-root', type=Path, required=True, help='One gsm_observed_<hash> profile')
    parser.add_argument('--start', required=True, help='Inclusive UTC date/time')
    parser.add_argument('--end', required=True, help='Exclusive UTC date/time')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing results are never overwritten')
    parser.add_argument('--spedas-dir', type=Path, default=Path('/mnt/j/observation_data'))
    parser.add_argument('--species', nargs='+', choices=SPECIES, default=list(SPECIES))
    parser.add_argument('--sectors', nargs='+', choices=('night', 'day'), default=['night', 'day'])
    parser.add_argument('--min-count', type=int, default=100)
    parser.add_argument('--min-days', type=int, default=1)
    parser.add_argument('--max-orbit-gap', type=float, default=120.)
    parser.add_argument('--vmax-kms', type=float, help='Optional fixed limit, shared by all panels and species')
    parser.add_argument('--dry-run', action='store_true', help='Inventory only, no NPZ/CDF reads')
    args = parser.parse_args(argv)
    start, end = parse(args.start), parse(args.end)
    if end <= start or args.min_count < 1 or args.min_days < 1 or not np.isfinite(args.max_orbit_gap) or args.max_orbit_gap <= 0:
        parser.error('Invalid period, count, or interpolation gap')
    if args.vmax_kms is not None and (not np.isfinite(args.vmax_kms) or args.vmax_kms <= 0):
        parser.error('--vmax-kms must be finite and positive')
    profile_path = args.gsm_root / 'profile.json'
    profile = json.loads(profile_path.read_text())
    config = profile['config']
    if config.get('schema') != 'lepi_gsm_observed_velocity_v1' or config.get('units') != 'km/s':
        raise ValueError('Expected GSM observed-velocity profile in km/s')
    items, state_counts = [], Counter()
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor < end:
        month = args.gsm_root / cursor.strftime('%Y/%m')
        for path in sorted((month / 'state').glob('*/*.json')):
            state = json.loads(path.read_text())
            lo = parse(state['start_utc'])
            hi = parse(state.get('end_utc_exclusive', state['start_utc']))
            if lo >= end or (hi <= start and lo < start):
                continue
            state_counts[state['status']] += 1
            if state['status'] in ('complete', 'partial'):
                if state['profile_hash'] != profile['profile_hash']:
                    raise ValueError('Mixed GSM profiles: ' + str(path))
                items.append((lo, path, month / 'data' / path.parent.name / (path.stem + '.npz'), state))
        cursor = (cursor.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    items.sort()
    report = {'status': 'inventory', 'start_utc': start.isoformat(), 'end_utc_exclusive': end.isoformat(),
              'gsm_root': str(args.gsm_root.resolve()), 'profile': profile, 'state_counts': dict(state_counts),
              'scope': 'Existing GSM states only; does not establish DSI/GSM period completeness',
              'estimator': 'Equal-weight native records per species; mean(v) and mean(abs(v)); no smoothing or clipping',
              'l_coordinate': 'Definitive ORB pos_Lm[:,0], 90 degree pitch angle (same as KAW maps)',
              'latitude_coordinate': 'Definitive ORB pos_rmlatmlt[:,1], signed MLAT in degrees',
              'mlt_coordinate': 'Definitive ORB pos_rmlatmlt[:,2], shortest-arc interpolation',
              'orbit_boundary_policy': 'Each daily CDF owns [day 00:00, next day 00:00); adjacent daily endpoint overlaps discarded',
              'display': 'Dipole schematic from L and MLAT, not a GSM spatial plane',
              'l_edges': L_EDGES.tolist(), 'mlat_edges_deg': LAT_EDGES.tolist(), 'mlt_edges_hour': MLT_EDGES.tolist(),
              'l_mlt_latitude_selection': [-50., 50.], 'l_mlt_sector_selection': args.sectors,
              'display_orientation': 'Earth white left / black right; L-MLAT dayside x-axis reversed without changing velocity signs; L-MLT 00 right, 06 top, 12 left, 18 bottom',
              'species': args.species, 'sectors': args.sectors, 'min_count': args.min_count, 'min_days': args.min_days,
              'max_orbit_gap_seconds': args.max_orbit_gap, 'vmax_kms': args.vmax_kms,
              'quality': 'moment_valid & rotation_valid & species velocity_valid & all finite components; legacy complete without moment_valid assumes all source moments valid; partial without moment_valid rejected; no extra scientific flags',
              'sources': {str(Path(__file__).resolve()): sha(__file__), str(Path(__file__).with_name('core.py').resolve()): sha(Path(__file__).with_name('core.py'))}}
    if args.dry_run:
        print(json.dumps(report, indent=2)); return 0
    if not items:
        raise ValueError('No usable GSM states in requested period')
    if args.output_dir.exists():
        raise FileExistsError('Choose a new --output-dir: ' + str(args.output_dir))
    os.nice(19)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=args.output_dir.name + '.partial-', dir=args.output_dir.parent))
    print(f'Writing staged results: {work}', flush=True)
    accs = {(s, sec): Accumulator() for s in args.species for sec in args.sectors}
    mlt_accs = {s: Accumulator(L_EDGES, MLT_EDGES) for s in args.species}
    diagnostics = Counter()
    orbit_inputs, loaded_day, last_time = {}, None, None
    try:
        with (work / 'inputs.jsonl').open('w') as provenance:
            for number, (lo, state_path, path, state) in enumerate(items, 1):
                day = lo.replace(hour=0, minute=0, second=0, microsecond=0)
                if loaded_day != day:
                    support = []
                    for offset in (-1, 0, 1):
                        support_day = day + dt.timedelta(days=offset)
                        try:
                            op = orbit_path(args.spedas_dir, support_day)
                        except FileNotFoundError:
                            if offset == 0:
                                raise
                            continue  # No extrapolation at missing day boundaries.
                        before = stamp(op)
                        support_times, support_values = read_orbit(op)
                        owned = ((support_times >= support_day.timestamp()) &
                                 (support_times < (support_day + dt.timedelta(days=1)).timestamp()))
                        support.append((support_times[owned], support_values[owned]))
                        digest = sha(op)
                        if stamp(op) != before:
                            raise RuntimeError('Orbit changed during read: ' + str(op))
                        orbit_inputs[str(op.resolve())] = digest
                    joined_t = np.concatenate([x[0] for x in support])
                    joined_v = np.concatenate([x[1] for x in support])
                    order = np.argsort(joined_t, kind='stable')
                    ot, ov = series(joined_t[order], joined_v[order])
                    loaded_day = day
                before = stamp(path)
                if sha(path) != state['output_sha256']:
                    raise ValueError('GSM NPZ hash mismatch: ' + str(path))
                with np.load(path, allow_pickle=False) as data:
                    manifest = json.loads(str(data['manifest_json']))
                    if manifest['profile_hash'] != profile['profile_hash'] or manifest['config'] != config:
                        raise ValueError('GSM manifest/profile mismatch')
                    times = data['time_unix']
                    if times.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                        raise ValueError('Invalid GSM times')
                    if np.any((times < lo.timestamp()) | (times >= parse(state['end_utc_exclusive']).timestamp())):
                        raise ValueError('GSM record outside state interval')
                    if last_time is not None and len(times) and times[0] <= last_time:
                        raise ValueError('Duplicate or overlapping GSM records')
                    if len(times):
                        last_time = times[-1]
                    in_period = (times >= start.timestamp()) & (times < end.timestamp())
                    lv, lat, mlt, orbit_valid = orbit_at(times, ot, ov, args.max_orbit_gap)
                    diagnostics['records_in_period'] += int(in_period.sum())
                    diagnostics['records_without_orbit'] += int((in_period & ~orbit_valid).sum())
                    if 'moment_valid' in data:
                        moment_valid = data['moment_valid']
                    elif state['status'] == 'complete':
                        moment_valid = np.ones(len(times), dtype=bool)
                        diagnostics['legacy_complete_records_without_moment_mask'] += int(in_period.sum())
                    else:
                        raise ValueError('Partial GSM input lacks moment_valid')
                    for valid_mask in (moment_valid, data['rotation_valid']):
                        if valid_mask.shape != times.shape or valid_mask.dtype.kind != 'b':
                            raise ValueError('Invalid GSM validity mask')
                    base = in_period & orbit_valid & moment_valid & data['rotation_valid']
                    for species in args.species:
                        velocity = data[species + '__velocity_gsm_observed']
                        if velocity.shape != (len(times), 3):
                            raise ValueError('Invalid velocity shape')
                        velocity_valid = data[species + '__velocity_valid']
                        if velocity_valid.shape != times.shape or velocity_valid.dtype.kind != 'b':
                            raise ValueError('Invalid species validity mask')
                        good = base & velocity_valid & np.isfinite(velocity).all(axis=1)
                        diagnostics[species + '_valid_records_before_mesh'] += int(good.sum())
                        chosen_sector = np.zeros(len(times), dtype=bool)
                        for sector in args.sectors:
                            chosen_sector |= sector_mask(mlt, sector)
                            count = accs[species, sector].add(lv, lat, velocity, good & sector_mask(mlt, sector), day.date().isoformat())
                            diagnostics[species + '_' + sector + '_binned_records'] += count
                        latitude_selection = (lat >= LAT_EDGES[0]) & (lat <= LAT_EDGES[-1])
                        count = mlt_accs[species].add(lv, mlt, velocity, good & chosen_sector & latitude_selection, day.date().isoformat())
                        diagnostics[species + '_l_mlt_binned_records'] += count
                if stamp(path) != before or json.loads(state_path.read_text()) != state:
                    raise RuntimeError('GSM input changed during aggregation')
                provenance.write(json.dumps({'state': str(state_path.resolve()), 'state_sha256': sha(state_path),
                                             'npz': str(path.resolve()), 'npz_sha256': state['output_sha256']}) + '\n')
                if number == 1 or number % 1000 == 0 or number == len(items):
                    print(f'{number}/{len(items)} GSM chunks', flush=True)
        for op, digest in orbit_inputs.items():
            if sha(op) != digest:
                raise RuntimeError('Orbit changed during aggregation: ' + op)
        # Share color scales across species, sectors and components.
        limits = [0., 0.]
        for acc in [*accs.values(), *mlt_accs.values()]:
            visible = (acc.n >= args.min_count) & (acc.days >= args.min_days)
            for row, values in enumerate(acc.means()):
                finite = np.abs(values[visible])
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    limits[row] = max(limits[row], float(finite.max()))
        limits = [args.vmax_kms or value or 1. for value in limits]
        report['color_limits_kms'] = {'signed': [-limits[0], limits[0]], 'absolute': [0., limits[1]]}
        period = f'{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC (end exclusive)'
        for (species, sector), acc in accs.items():
            stem = work / (species + '_' + sector)
            save_table(acc, stem.with_suffix('.csv'))
            plot(acc, SPECIES[species], sector, period, stem, args.min_count, args.min_days, limits, args.vmax_kms is not None)
        for species, acc in mlt_accs.items():
            stem = work / (species + '_l_mlt')
            save_table(acc, stem.with_suffix('.csv'), coordinate='mlt')
            plot(acc, SPECIES[species], 'all', period, stem, args.min_count, args.min_days, limits,
                 args.vmax_kms is not None, coordinate='l_mlt')
        report.update(status='complete', diagnostics=dict(diagnostics), orbit_inputs_sha256=orbit_inputs,
                      inputs_manifest_sha256=sha(work / 'inputs.jsonl'),
                      created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (work / 'provenance.json').write_text(json.dumps(report, indent=2))
        work.rename(args.output_dir)
    except BaseException:
        (work / 'INCOMPLETE.txt').write_text('Aggregation failed or interrupted; directory is not a published result.\n')
        raise
    print(args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
