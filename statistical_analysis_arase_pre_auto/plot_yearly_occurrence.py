#!/usr/bin/env python3
"""Plot Arase dwelling time, KAW event observation time and occurrence probability."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

import batch_arase as batch
import local_orbit_dwelling as orbit
import plot_occurrence_probability as occurrence
import plot_kappa_x0_statistics as spatial
import statistics_phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period
from statistics_plotting import add_output_arguments, should_write


COORDINATES = ("l_mlat", "l_abs_mlat", "l_mlt")


def range_table(phase, coordinate, min_occupied_blocks=6):
    selected, _ = spatial._spatial_rows(phase, coordinate)
    selected["_accepted_event"] = selection.accepted_events(selected, min_occupied_blocks)
    spec = spatial.POSITION_COLUMNS[coordinate]
    le = spatial.PLOT_CONFIG["l_edges"]
    ce = spatial.PLOT_CONFIG[spec["edges_key"]]
    rows = []
    for i in range(len(le) - 1):
        for j in range(len(ce) - 1):
            group = selected.loc[selected._l_bin.eq(i) & selected._coordinate_bin.eq(j)]
            fitted = group.loc[group["_accepted_event"]]
            rows.append({
                "phase_mode": phase.phase_mode.iloc[0], "coordinate_system": coordinate,
                "l_bin": i, "coordinate_bin": j,
                "l_lower": le[i], "l_upper": le[i + 1],
                "coordinate_lower": ce[j], "coordinate_upper": ce[j + 1],
                "n_batch_ranges": len(group), "n_kaw_ranges": len(fitted),
                "batch_range_duration_sec": float(group.range_duration_sec.sum()),
                "kaw_duration_sec": float(fitted.range_duration_sec.sum()),
                "n_fit_north": int(fitted.range_MLAT_deg_mean.gt(0).sum()),
                "n_fit_south": int(fitted.range_MLAT_deg_mean.lt(0).sum()),
                "n_fit_equator": int(fitted.range_MLAT_deg_mean.eq(0).sum()),
            })
    return pd.DataFrame(rows)


def plot_counts(table, phase, coordinate, date_label, output_base, maximum):
    le = spatial.PLOT_CONFIG["l_edges"]
    ce = spatial.PLOT_CONFIG[spatial.POSITION_COLUMNS[coordinate]["edges_key"]]
    shape = (len(le) - 1, len(ce) - 1)
    polar = coordinate == "l_mlt"
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=occurrence.PLOT_CONFIG["dpi"],
                            subplot_kw={"projection": "polar"} if polar else {})
    cmap = plt.get_cmap("viridis").with_extremes(bad="0.82")
    for ax, column, title in zip(axes, ["n_batch_ranges", "n_kaw_ranges"],
                                 ["Analyzed ranges", "KAW events (accepted ranges)"]):
        values = table[column].to_numpy(float).reshape(shape)
        values = np.ma.masked_where(values <= 0, values)
        if polar:
            mesh = ax.pcolormesh(2 * np.pi * ce / 24, le, values,
                                 cmap=cmap, norm=LogNorm(1, max(maximum, 2)), shading="flat")
            spatial._configure_polar_axis(ax, le)
        else:
            xe, ze = occurrence._dipole_mesh(le, ce)
            mesh = ax.pcolormesh(xe, ze, values.T, cmap=cmap,
                                 norm=LogNorm(1, max(maximum, 2)), shading="flat")
            spatial._add_meridional_grid(ax, le, ce, coordinate)
            occurrence.add_meridional_earth(ax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.15, float(np.max(xe)) * 1.04)
            zl = float(np.max(np.abs(ze))) * 1.06
            ax.set_ylim(-1.15 if coordinate == "l_abs_mlat" else -zl, zl)
            ax.set_xlabel(r"Nightside distance [$R_{\mathrm{E}}$]")
        ax.set_title(f"{title}\nN = {int(table[column].sum())}", pad=23)
        fig.colorbar(mesh, ax=ax, label="Number of ranges", pad=.12 if polar else .025)
    if not polar:
        axes[0].set_ylabel(spatial._meridional_ylabel(coordinate))
    fig.suptitle(f"Arase range counts in {spatial._coordinate_title(coordinate)}: "
                 f"{statistics_phases.PHASES[phase]}\n{date_label}")
    fig.tight_layout(rect=(0, 0, 1, .88))
    return occurrence._save_figure(fig, output_base)


def dwelling_tables(files):
    """Sum native-cadence ORB residence times before forming ratios."""
    sums = {}
    total_sec, total_samples = 0., 0
    for day, path in sorted(files.items()):
        points = orbit.read_orbit_day(path, day, occurrence.PLOT_CONFIG["orbit_gap_threshold_sec"])
        total_sec += float(points.dwell_sec.sum())
        total_samples += len(points)
        for coordinate in COORDINATES:
            duration, samples, _ = occurrence.aggregate_dwelling(points, coordinate)
            if coordinate not in sums:
                sums[coordinate] = [duration, samples]
            else:
                sums[coordinate][0] += duration
                sums[coordinate][1] += samples
    return sums, {"orbit_samples": total_samples, "orb_weighted_duration_sec": total_sec}


def attach_dwelling(table, durations, samples):
    result = table.copy()
    result["dwelling_duration_sec"] = durations.ravel()
    result["n_dwelling_samples"] = samples.ravel()
    if (result.kaw_duration_sec > result.dwelling_duration_sec + 1e-6).any():
        raise ValueError("KAW event interval time exceeds ORB dwelling time; representative-bin approximation is inconsistent")
    result["dwelling_duration_hour"] = result.dwelling_duration_sec / 3600
    result["kaw_duration_hour"] = result.kaw_duration_sec / 3600
    result["occurrence_probability"] = np.divide(
        result.kaw_duration_sec, result.dwelling_duration_sec,
        out=np.full(len(result), np.nan), where=result.dwelling_duration_sec > 0)
    result["occurrence_percent"] = 100 * result.occurrence_probability
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    add_output_arguments(parser)
    selection.add_selection_argument(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Content parent; default period/min_blocks_N")
    parser.add_argument("--phase", action="append", choices=tuple(statistics_phases.PHASES))
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--download-missing-orbit", action="store_true")
    args = parser.parse_args(argv)
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    source = root / "inputs" / "provenance.json"
    master_path = root / "inputs" / "arase_phase_master.csv"
    master, provenance = selection.load_prepared_inputs(root, period)
    master = statistics_phases.classify_phases(master)
    output_root = args.output_dir or root / selection.selection_key(args.min_occupied_blocks)
    phases = {p: occurrence.select_phase_ranges(master, p)
              for p in dict.fromkeys(args.phase or statistics_phases.PHASES)}
    written = []
    months = [s["month"] for s in provenance["sources"] if s["status"] == "processed"]
    excluded_months = [s["month"] for s in provenance["sources"] if s["status"] == "no_data"]
    label = f"{period.start:%Y-%m-%d} to {period.last_day:%Y-%m-%d}"
    if excluded_months:
        label += "; excluded: " + ", ".join(excluded_months)
    label += f"; occupied 10 s blocks >= {args.min_occupied_blocks}"
    tables = {(p, c): range_table(f, c, args.min_occupied_blocks) for p, f in phases.items() for c in COORDINATES}
    maximum = max(int(t.n_batch_ranges.max()) for t in tables.values())
    if args.counts_only:
        count_dir = output_root / "event_counts"
        count_dir.mkdir(parents=True, exist_ok=True)
        for (phase, coordinate), table in tables.items():
            base = count_dir / f"arase_range_counts_{coordinate}_{phase}_{period.date_suffix}"
            if not should_write(base, overwrite=not args.skip_existing):
                continue
            table.to_csv(base.with_suffix(".csv"), index=False)
            plot_counts(table, phase, coordinate, label, base, maximum)
            written.append(str(base))
            print(f"{phase}/{coordinate}: analyzed={table.n_batch_ranges.sum()}, fitted={table.n_kaw_ranges.sum()}", flush=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master": str(master_path), "master_sha256": batch.file_sha256(master_path),
        "source_provenance_sha256": batch.file_sha256(source),
        "selected_months": months, "excluded_months": excluded_months,
        "selection": selection.selection_metadata(master, args.min_occupied_blocks),
        "event_unit": "accepted KAW analysis range; parent events may contain multiple ranges",
        "event_selection": f"phase status == ok and n_occupied_blocks >= {args.min_occupied_blocks}",
        "numerator": "sum of full KAW event interval durations, not detected sample durations",
        "denominator": "ORB dwelling time during processed months, without instrument-availability mask",
        "plot_config": occurrence.PLOT_CONFIG,
        "code_sha256": {str(Path(m.__file__).resolve()): batch.file_sha256(Path(m.__file__))
                        for m in [orbit, occurrence, spatial, statistics_phases, selection]},
    }
    metadata["code_sha256"][str(Path(__file__).resolve())] = batch.file_sha256(Path(__file__))
    if args.counts_only:
        if written:
            metadata["written"] = written
            (count_dir / "provenance.json").write_text(json.dumps(metadata, indent=2, default=lambda v: v.tolist()) + "\n")
        return 0
    files, missing = orbit.orbit_inventory(months, args.spedas_dir)
    print(f"ORB selected days={len(files) + len(missing)}, local={len(files)}, missing={len(missing)}", flush=True)
    if missing:
        if not args.download_missing_orbit:
            raise FileNotFoundError(f"Missing {len(missing)} ORB days; probability not calculated. Use --download-missing-orbit to acquire them.")
        files.update(orbit.download_missing_days(missing, args.spedas_dir))
    records = [{"day": str(d.date()), "path": str(p), "sha256": batch.file_sha256(p)} for d, p in sorted(files.items())]
    denominator_config = {key: occurrence.PLOT_CONFIG[key] for key in [
        "l_edges", "mlat_edges_deg", "abs_mlat_edges_deg", "mlt_edges_hour",
        "l_mlat_mlt_sector_hour", "orbit_gap_threshold_sec"]}
    signature_payload = {"files": records, "months": months, "config": denominator_config,
                         "loader_sha256": batch.file_sha256(Path(orbit.__file__)),
                         "binning_sha256": batch.file_sha256(Path(occurrence.__file__))}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, default=lambda v: v.tolist()).encode()).hexdigest()
    output = output_root / "occurrence_probability"
    output.mkdir(parents=True, exist_ok=True)
    cache = root / "cache" / "orbit_dwelling_bins.npz"
    sums = None
    cache_matches = False
    for candidate in [cache, root / "occurrence_probability" / "orbit_dwelling_bins.npz"]:
        if candidate.exists():
            with np.load(candidate, allow_pickle=False) as data:
                if str(data["signature"]) == signature:
                    sums = {c: [data[c + "_seconds"], data[c + "_samples"]] for c in COORDINATES}
                    metadata.update(json.loads(str(data["coverage_json"])))
                    cache_matches = candidate == cache
                    break
    if sums is None:
        sums, coverage = dwelling_tables(files)
        metadata.update(coverage)
    if not cache_matches:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {c + suffix: values[i] for c, values in sums.items()
                  for i, suffix in enumerate(["_seconds", "_samples"])}
        coverage = {key: metadata[key] for key in ["orbit_samples", "orb_weighted_duration_sec"]}
        np.savez_compressed(cache, signature=signature, coverage_json=json.dumps(coverage), **arrays)
    metadata.update(orbit_files=records, dwelling_cache_signature=signature)
    expected_seconds = sum((pd.Timestamp(m + "-01") + pd.offsets.MonthBegin(1)
                            - pd.Timestamp(m + "-01")).total_seconds() for m in months)
    metadata["selected_calendar_duration_sec"] = expected_seconds
    # Shared time scale across all phase/coordinate panels; percent upper bound remains 100%.
    probability_tables = {key: attach_dwelling(t, *sums[key[1]]) for key, t in tables.items()}
    times = np.concatenate([t[["dwelling_duration_hour", "kaw_duration_hour"]].to_numpy().ravel() for t in probability_tables.values()])
    probabilities = np.concatenate([t.occurrence_percent.to_numpy() for t in probability_tables.values()])
    occurrence.PLOT_CONFIG["time_log_limits_hour"] = occurrence._log_bounds(times, (None, None))
    occurrence.PLOT_CONFIG["probability_log_limits_percent"] = occurrence._log_bounds(probabilities, (None, 100.))
    metadata["plot_config"] = occurrence.PLOT_CONFIG
    for (phase, coordinate), table in probability_tables.items():
        base = output / f"arase_kaw_occurrence_{coordinate}_{phase}_{period.date_suffix}"
        if not should_write(base, overwrite=not args.skip_existing):
            continue
        table.to_csv(base.with_suffix(".csv"), index=False)
        plotter = occurrence.plot_l_mlt if coordinate == "l_mlt" else occurrence.plot_l_mlat
        plotter(table, phase, label, base)
        written.append(str(base))
    if written:
        metadata["written"] = written
        (output / "provenance.json").write_text(json.dumps(metadata, indent=2, default=lambda v: v.tolist()) + "\n")
    print(f"ORB coverage: {metadata['orb_weighted_duration_sec']/3600:.3f}/{expected_seconds/3600:.3f} hours", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
