#!/usr/bin/env python3
"""Range-count histograms of saved log10(X0) and kappa, separated by phase."""

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
import plot_ae_parameter_statistics as parameters
import statistics_phases as phases
import statistics_selection as selection
import statistics_plotting as styling
from statistics_period import add_period_arguments, resolve_period


PLOT_CONFIG = {"figure_size": (17, 10), "dpi": 300, "font_size": 13}
EXTENSIONS = (".png", ".pdf", ".csv", ".points.csv")


def histogram_points(master, phase):
    frame = master.loc[master.phase_mode.eq(phase)]
    if frame.range_uid.isna().any() or frame.range_uid.duplicated().any():
        raise ValueError("Histograms require unique nonmissing range_uid within each phase")
    points = []
    for family, spec in parameters.FAMILIES.items():
        for component, column in spec["columns"].items():
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = np.isfinite(values)
            saved = frame.loc[valid, ["range_uid", "source_phase_mode", "phase_mode"]].copy()
            saved["family"], saved["component"], saved["source_column"] = family, component, column
            saved["value"] = values.loc[valid]
            points.append(saved)
    return pd.concat(points, ignore_index=True)


def make_edges(values, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Histogram bin width must be finite and positive")
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([0., width])
    first = int(np.floor(values.min() / width))
    last = int(np.floor(values.max() / width)) + 1
    if last - first > 10000:
        raise ValueError("Histogram bin width creates more than 10000 bins")
    return np.arange(first, last + 1, dtype=float) * width


def aggregate(points, edges, n_ranges):
    rows = []
    for family, spec in parameters.FAMILIES.items():
        for component, column in spec["columns"].items():
            values = points.loc[points.family.eq(family) & points.component.eq(component), "value"].to_numpy(float)
            boundary = edges[family, component]
            counts, _ = np.histogram(values, bins=boundary)
            if counts.sum() != len(values):
                raise ValueError("Histogram edges do not cover all finite samples")
            median = float(np.median(values)) if len(values) else np.nan
            variance = float(np.var(values, ddof=1)) if len(values) >= 2 else np.nan
            for index, count in enumerate(counts):
                rows.append({"family": family, "component": component, "source_column": column,
                             "bin": index, "lower": boundary[index], "upper": boundary[index + 1],
                             "center": (boundary[index] + boundary[index + 1]) / 2,
                             "count": int(count), "total_n": len(values),
                             "n_ranges": n_ranges, "n_nonfinite": n_ranges - len(values),
                             "median": median, "variance": variance, "std": np.sqrt(variance),
                             "variance_ddof": 1, "variance_unit": "dex^2" if family == "log10_x0" else "dimensionless"})
    return pd.DataFrame(rows)


def plot(table, phase, edges, limits, label, base):
    with plt.rc_context({"font.size": PLOT_CONFIG["font_size"]}):
        fig, axes = plt.subplots(2, 3, figsize=PLOT_CONFIG["figure_size"])
        for row, (family, spec) in enumerate(parameters.FAMILIES.items()):
            for col, component in enumerate(parameters.COMPONENTS):
                ax = axes[row, col]
                group = table.loc[table.family.eq(family) & table.component.eq(component)]
                boundary = edges[family, component]
                color = parameters.PLOT_CONFIG["component_colors"][component]
                ax.stairs(group["count"].to_numpy(), boundary, fill=True, color=color, alpha=.6, lw=1.)
                ax.set_xlim(boundary[0], boundary[-1])
                ax.set_ylim(0, limits[family, component])
                ax.set_xlabel(spec["label"](component))
                ax.set_ylabel("Number of ranges")
                median, variance = group['median'].iloc[0], group.variance.iloc[0]
                if np.isfinite(median):
                    ax.axvline(median, color="black", ls="--", lw=1.5)
                median_text = f"{median:.3g}" if np.isfinite(median) else "n/a"
                variance_text = f"{variance:.3g}" if np.isfinite(variance) else "n/a"
                unit = r" dex$^2$" if family == "log10_x0" else ""
                ax.set_title(f"({chr(97 + row * 3 + col)}) {component}; N = {int(group.total_n.iloc[0])}\n"
                             + f"median = {median_text}; " + rf"$s^2$ = {variance_text}" + unit)
                styling.detail_grid(ax)
        fig.suptitle(f"Arase KAW parameter histograms: {phases.PHASES[phase]} (all MLT)\n{label}\n"
                     + r"Dashed line: median; $s^2$: sample variance (ddof=1)")
        fig.tight_layout(rect=(0, 0, 1, .89))
        for suffix in (".png", ".pdf"):
            styling.save_figure(fig, base.with_suffix(suffix), dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    styling.add_output_arguments(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Content directory (kappa_x0_histograms)")
    parser.add_argument("--phase", action="append", choices=tuple(phases.PHASES))
    parser.add_argument("--log-x0-bin-width-dex", type=float, default=.5)
    parser.add_argument("--kappa-bin-width", type=float, default=.2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    widths = {"log10_x0": args.log_x0_bin_width_dex, "kappa": args.kappa_bin_width}
    if any(not np.isfinite(width) or width <= 0 for width in widths.values()):
        raise ValueError("Histogram bin widths must be finite and positive")
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks) / "kappa_x0_histograms"
    master, provenance = selection.load_prepared_inputs(root, period)
    master = phases.classify_phases(master)
    selected = selection.select_events(master, args.min_occupied_blocks)
    requested = list(dict.fromkeys(args.phase or phases.PHASES))
    frames = {phase: histogram_points(selected, phase) for phase in requested}
    pooled = pd.concat(frames.values(), ignore_index=True)
    if pooled.empty:
        raise ValueError("No finite values in requested histogram samples")
    edges = {(family, component): make_edges(
        pooled.loc[pooled.family.eq(family) & pooled.component.eq(component), "value"], widths[family])
        for family in parameters.FAMILIES for component in parameters.COMPONENTS}
    tables = {phase: aggregate(points, edges, int(selected.phase_mode.eq(phase).sum()))
              for phase, points in frames.items()}
    # Common bin edges and panel-specific count limits across requested phases.
    limits = {(family, component): max(1., 1.1 * max(
        t.loc[t.family.eq(family) & t.component.eq(component), "count"].max() for t in tables.values()))
        for family in parameters.FAMILIES for component in parameters.COMPONENTS}
    label = (f"{period.start:%Y-%m-%d} to {period.last_day:%Y-%m-%d}; "
             f"occupied 10 s blocks >= {args.min_occupied_blocks}; "
             f"bins: log10 X0 {args.log_x0_bin_width_dex:g} dex, kappa {args.kappa_bin_width:g}")
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for phase, table in tables.items():
        if frames[phase].empty:
            print(f"Skip empty phase: {phase}", flush=True)
            continue
        base = output / f"arase_kappa_x0_histograms_{phase}_{period.date_suffix}"
        if not styling.should_write(base, overwrite=not args.skip_existing, extensions=EXTENSIONS):
            continue
        table.insert(0, "phase_mode", phase)
        table.to_csv(base.with_suffix(".csv"), index=False)
        frames[phase].to_csv(base.with_suffix(".points.csv"), index=False)
        plot(table, phase, edges, limits, label, base)
        written.extend(str(base.with_suffix(s)) for s in EXTENSIONS)
        print(f"Wrote histogram: {phase}", flush=True)
    if written:
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
                    "master_sha256": provenance["annual_master_sha256"],
                    "source_provenance_sha256": batch.file_sha256(root / "inputs/provenance.json"),
                    "selection": selection.selection_metadata(master, args.min_occupied_blocks),
                    "mlt_sector": "all", "valid_fraction_screening": False,
                    "definition": "one range per finite value; independent finite selection for each parameter; raw counts, not density; no minimum bin count",
                    "summary": "median, sample variance (ddof=1) and sample std from individual finite range values, not bin centers; log10_X0 variance in dex^2; variance/std undefined for n<2",
                    "bin_widths": widths, "grid_config": styling.GRID_CONFIG, "plot_config": PLOT_CONFIG,
                    "panels": [{"family": f, "component": c, "edges": e.tolist(), "count_ylim": [0, limits[f, c]]}
                               for (f, c), e in edges.items()],
                    "written": written,
                    "code_sha256": {str(Path(p).resolve()): batch.file_sha256(Path(p)) for p in
                                    (__file__, parameters.__file__, phases.__file__, selection.__file__, styling.__file__)}}
        (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
