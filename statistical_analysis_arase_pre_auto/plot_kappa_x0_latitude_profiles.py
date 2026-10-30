#!/usr/bin/env python3
"""Plot range values and latitude-bin distributions separately for each L bin."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import batch_arase as batch
import plot_kappa_x0_statistics as spatial
import statistics_phases as phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period
from statistics_plotting import detail_grid, add_output_arguments, should_write
import statistics_plotting
import statistics_correlations as correlations


PLOT_CONFIG = {
    "figure_size": (16, 9), "dpi": 300, "font_size": 13,
    "raw_marker_size": 15, "raw_alpha": .25,
    "colors": {"E": "tab:red", "B": "tab:blue", "S": "tab:green"},
}
COORDINATES = ("l_mlat", "l_abs_mlat")


def profile_data(frame, phase, coordinate):
    if coordinate not in COORDINATES:
        raise ValueError(f"Unknown latitude coordinate: {coordinate}")
    points, diagnostic = spatial._spatial_rows(frame, coordinate)
    latitude = pd.to_numeric(points["range_MLAT_deg_mean"], errors="coerce")
    points["_latitude"] = latitude.abs() if coordinate == "l_abs_mlat" else latitude
    tables = [spatial.aggregate_statistics(frame, phase, coordinate, family)[0]
              for family in ("log10_x0", "kappa")]
    table = pd.concat(tables, ignore_index=True)
    table["latitude_center_deg"] = (table.coordinate_lower + table.coordinate_upper) / 2
    coefficients = []
    for index in table.l_bin.unique():
        cell = points.loc[points._l_bin.eq(index)]
        for family, spec in spatial.FAMILIES.items():
            for component, column in spec["columns"].items():
                values = pd.to_numeric(cell[column], errors="coerce")
                coefficients.append({"l_bin": index, "family": family, "component": component,
                                     **correlations.pair_correlations(cell._latitude, values),
                                     "pearson_x_coordinate": "abs_MLAT_deg" if coordinate == "l_abs_mlat" else "MLAT_deg",
                                     "pearson_y_coordinate": column})
    table = table.merge(pd.DataFrame(coefficients), on=["l_bin", "family", "component"],
                        how="left", validate="many_to_one")
    return points, table, diagnostic


def y_limits(frames):
    result = {}
    for family, spec in spatial.FAMILIES.items():
        result[family] = {}
        for component, column in spec["columns"].items():
            values = pd.concat([pd.to_numeric(f[column], errors="coerce") for f in frames], ignore_index=True)
            result[family][component] = spatial._limits(values)
    return result


def plot_profile(points, table, phase, coordinate, l_lower, l_upper, date_label,
                 output_base, minimum_bin_count=1, limits=None):
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    fig, axes = plt.subplots(2, 3, figsize=PLOT_CONFIG["figure_size"],
                             dpi=PLOT_CONFIG["dpi"], sharex=True)
    edges = spatial.PLOT_CONFIG[spatial.POSITION_COLUMNS[coordinate]["edges_key"]]
    for row, family in enumerate(("log10_x0", "kappa")):
        for col, component in enumerate(spatial.COMPONENTS):
            ax = axes[row, col]
            column = spatial.FAMILIES[family]["columns"][component]
            values = pd.to_numeric(points[column], errors="coerce")
            valid = np.isfinite(values)
            ax.scatter(points.loc[valid, "_latitude"], values.loc[valid],
                       s=PLOT_CONFIG["raw_marker_size"], alpha=PLOT_CONFIG["raw_alpha"],
                       color="0.4", edgecolors="none", label="range")
            group = table.loc[table.family.eq(family) & table.component.eq(component)].sort_values("coordinate_bin")
            shown = group.n.ge(minimum_bin_count) & np.isfinite(group["median"])
            color = PLOT_CONFIG["colors"][component]
            # Retain NaNs so a connecting line does not bridge unobserved bins.
            ax.plot(group.latitude_center_deg, group["median"].where(shown), color=color, lw=1.5)
            visible = group.loc[shown]
            ax.errorbar(visible.latitude_center_deg, visible["median"],
                        yerr=np.array([visible["median"] - visible.q16, visible.q84 - visible["median"]]),
                        fmt="o", color=color, capsize=3, ms=4, label="bin median [q16, q84]")
            for _, cell in visible.iterrows():
                ax.annotate(f"n={int(cell.n)}", (cell.latitude_center_deg, cell.q84),
                            xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, color=color)
            label = spatial.FAMILIES[family]["label"](component)
            ax.set_ylabel(label)
            ax.set_title(f"({chr(97 + row * 3 + col)}) {label}; N = {int(valid.sum())}\n"
                         + correlations.correlation_label(group.iloc[0].spearman_rho, group.iloc[0].pearson_r))
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xticks(np.arange(edges[0], edges[-1] + 1, 10))
            if limits is not None:
                ax.set_ylim(*limits[family][component])
            detail_grid(ax)
            if row == 1:
                ax.set_xlabel("|MLAT| [deg]" if coordinate == "l_abs_mlat" else "MLAT [deg]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    latitude_title = "|MLAT| (hemispheres pooled)" if coordinate == "l_abs_mlat" else "signed MLAT"
    fig.suptitle(f"Arase $X_0$ and $\\kappa$ versus {latitude_title}: {phases.PHASES[phase]}\n"
                 f"{l_lower:g} <= L < {l_upper:g}; 18-06 MLT; {date_label}")
    fig.tight_layout(rect=(0, .045, 1, .91))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    statistics_plotting.save_figure(fig, paths[0], dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    statistics_plotting.save_figure(fig, paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    add_output_arguments(parser)
    parser.add_argument("--statistics-dir", type=Path, help="Prepared period directory.")
    parser.add_argument("--output-dir", type=Path, help="Override the latitude-profile output directory.")
    parser.add_argument("--phase", action="append", choices=tuple(phases.PHASES))
    parser.add_argument("--coordinate", action="append", choices=COORDINATES)
    parser.add_argument("--l-bin", type=float, action="append", help="L-bin lower edge, e.g. 4 selects 4 <= L < 5; repeatable.")
    parser.add_argument("--min-bin-count", type=int, default=1)
    parser.add_argument("--independent-y-limits", action="store_true", help="Scale each figure separately; default: common per-parameter limits.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    period = resolve_period(args)
    if args.min_bin_count < 1:
        raise ValueError("--min-bin-count must be at least 1")
    le = spatial.PLOT_CONFIG["l_edges"]
    lower_edges = list(dict.fromkeys(args.l_bin or le[:-1].tolist()))
    if any(value not in le[:-1] for value in lower_edges):
        raise ValueError(f"--l-bin must be one of {le[:-1].tolist()}")
    indices = [int(np.flatnonzero(le == value)[0]) for value in lower_edges]
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks) / "kappa_x0_latitude_profiles"
    master, _ = selection.load_prepared_inputs(root, period)
    master = phases.classify_phases(master)
    selected = selection.select_events(master, args.min_occupied_blocks)
    requested_phases = list(dict.fromkeys(args.phase or phases.PHASES))
    coordinates = list(dict.fromkeys(args.coordinate or COORDINATES))
    data = {}
    for phase in requested_phases:
        frame = spatial._select_phase(selected, phase)
        for coordinate in coordinates:
            points, table, _ = profile_data(frame, phase, coordinate)
            data[phase, coordinate] = (
                points.loc[points._l_bin.isin(indices)].copy(),
                table.loc[table.l_bin.isin(indices)].copy())
    limits = None if args.independent_y_limits else y_limits([value[0] for value in data.values()])
    date_label = (f"{period.start:%Y-%m-%d} to {period.last_day:%Y-%m-%d}; "
                  f"occupied 10 s blocks >= {args.min_occupied_blocks}")
    written, diagnostics = [], []
    for (phase, coordinate), (points, table) in data.items():
        for index in indices:
            cell = points.loc[points._l_bin.eq(index)]
            diagnostic = {"phase": phase, "coordinate": coordinate,
                          "l_lower": float(le[index]), "l_upper": float(le[index + 1]),
                          "n_ranges": len(cell)}
            diagnostics.append(diagnostic)
            if cell.empty:
                print(f"Skip empty L bin: {diagnostic}", flush=True)
                continue
            tag = f"L_{le[index]:g}_{le[index + 1]:g}".replace(".", "p")
            base = output / coordinate / tag / f"arase_kappa_x0_latitude_profile_{phase}_{period.date_suffix}"
            if not should_write(base, overwrite=not args.skip_existing):
                continue
            base.parent.mkdir(parents=True, exist_ok=True)
            subset = table.loc[table.l_bin.eq(index)]
            subset.to_csv(base.with_suffix(".csv"), index=False)
            written.append(base.with_suffix(".csv"))
            written.extend(plot_profile(cell, subset, phase, coordinate, le[index], le[index + 1],
                                        date_label, base, args.min_bin_count, limits))
    if not written:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
        "master": str(root / "inputs/arase_phase_master.csv"),
        "master_sha256": batch.file_sha256(root / "inputs/arase_phase_master.csv"),
        "selection": selection.selection_metadata(master, args.min_occupied_blocks),
        "min_bin_count": args.min_bin_count, "independent_y_limits": args.independent_y_limits,
        "summary": "range-weighted median and q16-q84 distribution spread; not confidence interval",
        "plot_config": PLOT_CONFIG, "spatial_config": spatial.PLOT_CONFIG,
        "grid_config": statistics_plotting.GRID_CONFIG,
        "correlation": "Spearman rho and Pearson r per L bin, phase, component and family from individual finite range pairs; signed MLAT or abs(MLAT) as displayed, saved log10_X0 or kappa for y; no temporal or fit covariance correction; independent of min-bin-count display threshold",
        "diagnostics": diagnostics, "written": [str(path) for path in written],
        "code_sha256": {str(Path(m.__file__).resolve()): batch.file_sha256(Path(m.__file__))
                        for m in [spatial, selection, phases, statistics_plotting, correlations]},
    }
    metadata["code_sha256"][str(Path(__file__).resolve())] = batch.file_sha256(Path(__file__))
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2, default=lambda v: v.tolist()) + "\n")
    print(f"Wrote {len(written)} files: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
