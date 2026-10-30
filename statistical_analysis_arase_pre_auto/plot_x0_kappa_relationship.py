#!/usr/bin/env python3
"""Plot paired saved log10(X0) and kappa fits, one range per point."""

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
import plot_ae_parameter_statistics as ae
import statistics_phases as phases
import statistics_selection as selection
import statistics_plotting as styling
import statistics_correlations as correlations
from statistics_period import add_period_arguments, resolve_period


PLOT_CONFIG = {"figure_size": (17, 5.8), "dpi": 300, "font_size": 13}
EXTENSIONS = (".png", ".pdf", ".csv", ".points.csv")


def relationship_key(reference):
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference-kperp-rhoi must be finite and positive")
    return ("x0_kappa_relationship" if reference == 1. else
            "x0_kappa_relationship_kperp_rhoi_" + f"{reference:g}".replace(".", "p"))


def reference_points(points, reference):
    relationship_key(reference)
    frame = points.copy()
    frame["reference_kperp_rhoi"] = reference
    frame["log10_X_reference"] = frame.log10_X0 - frame.kappa * np.log10(reference)
    return frame


def paired_points(master, phase):
    """Require finite x/y from the same row and component; no coverage cut."""
    frame = master.loc[master.phase_mode.eq(phase)]
    rows = []
    for component in ae.COMPONENTS:
        x = pd.to_numeric(frame[f"log10_X0_{component}"], errors="coerce")
        y = pd.to_numeric(frame[f"kappa_{component}"], errors="coerce")
        keep = np.isfinite(x) & np.isfinite(y)
        points = frame.loc[keep, ["range_uid", "source_phase_mode", "phase_mode"]].copy()
        points["component"] = component
        points["log10_X0"] = x.loc[keep]
        points["kappa"] = y.loc[keep]
        rows.append(points)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["range_uid", "component"]).any():
        raise ValueError("Duplicate range_uid/component pairs within phase")
    return result


def make_edges(values, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("X0 bin width must be finite and positive")
    first = int(np.floor(values.min() / width))
    last = int(np.floor(values.max() / width)) + 1
    if last - first > 10000:
        raise ValueError("X0 bin width creates more than 10000 bins")
    return np.arange(first, last + 1, dtype=float) * width


def aggregate(points, edges, minimum_count, x_column="log10_X0"):
    rows = []
    for component in ae.COMPONENTS:
        group = points.loc[points.component.eq(component)]
        x, y = group[x_column].to_numpy(), group.kappa.to_numpy()
        correlation = correlations.pair_correlations(x, y)
        bins = np.searchsorted(edges[component], x, side="right") - 1
        for index, (lower, upper) in enumerate(zip(edges[component][:-1], edges[component][1:])):
            stats = ae._summary(y[bins == index])
            rows.append({"component": component, "bin": index, f"lower_{x_column}": lower,
                         f"upper_{x_column}": upper, f"center_{x_column}": (lower + upper) / 2,
                         **stats, "shown": stats["n"] >= minimum_count,
                         "overall_n": len(x), **correlation,
                         "pearson_x_coordinate": x_column, "pearson_y_coordinate": "kappa"})
    return pd.DataFrame(rows)


def plot(points, table, phase, edges, limits, label, base, reference=1.):
    x_column = "log10_X0" if reference == 1. else "log10_X_reference"
    center_column = "center_" + x_column
    with plt.rc_context({"font.size": PLOT_CONFIG["font_size"]}):
        fig, axes = plt.subplots(1, 3, figsize=PLOT_CONFIG["figure_size"])
        for ax, component in zip(axes, ae.COMPONENTS):
            group = points.loc[points.component.eq(component)]
            bins = table.loc[table.component.eq(component)]
            shown = bins.shown & np.isfinite(bins["median"])
            visible = bins.loc[shown]
            color = ae.PLOT_CONFIG["component_colors"][component]
            ax.scatter(group[x_column], group.kappa, s=22, alpha=.28,
                       color="0.35", edgecolors="none", label="range")
            ax.plot(bins[center_column], bins["median"].where(shown), color=color, lw=1.8)
            ax.errorbar(visible[center_column], visible["median"],
                        yerr=np.array([visible["median"] - visible.q16, visible.q84 - visible["median"]]),
                        fmt="o", color=color, ms=5, capsize=3, label="bin median [q16, q84]")
            for _, cell in visible.iterrows():
                ax.annotate(f"n={int(cell.n)}", (cell[center_column], cell["median"]),
                            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xlabel(ae.FAMILIES["log10_x0"]["label"](component) if reference == 1. else
                          rf"$\log_{{10}} X_{{\mathrm{{{component}}}}}(k_\perp\rho_{{\mathrm{{i}}}}={reference:g})$")
            ax.set_ylabel(ae.FAMILIES["kappa"]["label"](component))
            ax.set_title(f"{component}; N = {len(group)}\n"
                         + correlations.correlation_label(bins.iloc[0].spearman_rho, bins.iloc[0].pearson_r))
            ax.set_xlim(edges[component][0], edges[component][-1])
            ax.set_ylim(*limits["kappa"][component])
            styling.detail_grid(ax)
        amplitude = "X0" if reference == 1. else rf"$X(k_\perp\rho_{{\mathrm{{i}}}}={reference:g})$"
        fig.suptitle(f"Arase KAW {amplitude}-kappa: {phases.PHASES[phase]} (all MLT)\n{label}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, .07, 1, .86))
        for suffix in (".png", ".pdf"):
            styling.save_figure(fig, base.with_suffix(suffix), dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    styling.add_output_arguments(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Content directory (x0_kappa_relationship)")
    parser.add_argument("--phase", choices=tuple(phases.PHASES), action="append")
    parser.add_argument("--min-bin-count", type=int, default=5)
    parser.add_argument("--x0-bin-width-dex", type=float, default=.5)
    parser.add_argument("--reference-kperp-rhoi", type=float, default=1.,
                        help="Evaluate model amplitude at this k_perp*rho_i; default 1 retains original X0 plot")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.min_bin_count < 1 or not np.isfinite(args.x0_bin_width_dex) or args.x0_bin_width_dex <= 0:
        raise ValueError("Require positive bin width and min-bin-count >= 1")
    period = resolve_period(args)
    key = relationship_key(args.reference_kperp_rhoi)
    x_column = "log10_X0" if args.reference_kperp_rhoi == 1. else "log10_X_reference"
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks) / key
    master, provenance = selection.load_prepared_inputs(root, period)
    master = phases.classify_phases(master)
    selected = selection.select_events(master, args.min_occupied_blocks)
    requested = list(dict.fromkeys(args.phase or phases.PHASES))
    frames = {phase: paired_points(selected, phase) for phase in requested}
    if args.reference_kperp_rhoi != 1.:
        frames = {phase: reference_points(points, args.reference_kperp_rhoi) for phase, points in frames.items()}
    pooled = pd.concat(frames.values(), ignore_index=True)
    if pooled.empty:
        raise ValueError("No finite X0-kappa pairs for requested phases")
    # Share component-specific bins/limits across requested phase figures.
    edges = {component: make_edges(pooled.loc[pooled.component.eq(component), x_column], args.x0_bin_width_dex)
             if pooled.component.eq(component).any() else np.array([0., args.x0_bin_width_dex])
             for component in ae.COMPONENTS}
    limits = {"kappa": {}}
    for component in ae.COMPONENTS:
        values = pooled.loc[pooled.component.eq(component), "kappa"]
        if values.empty:
            limits["kappa"][component] = (0., 1.)
        else:
            low, high = float(values.min()), float(values.max())
            pad = max(.06 * (high - low), .05 if np.isclose(low, high) else 0.)
            limits["kappa"][component] = (low - pad, high + pad)
    output.mkdir(parents=True, exist_ok=True)
    label = (f"{period.start:%Y-%m-%d} to {period.last_day:%Y-%m-%d}; "
             f"occupied 10 s blocks >= {args.min_occupied_blocks}; "
             f"{'X0' if args.reference_kperp_rhoi == 1. else 'Amplitude'} bins {args.x0_bin_width_dex:g} dex; "
             f"bin n >= {args.min_bin_count}")
    written = []
    for phase, points in frames.items():
        if points.empty:
            print(f"Skip empty phase: {phase}", flush=True)
            continue
        base = output / f"arase_{key}_{phase}_{period.date_suffix}"
        if not styling.should_write(base, overwrite=not args.skip_existing, extensions=EXTENSIONS):
            continue
        table = aggregate(points, edges, args.min_bin_count, x_column=x_column)
        table.insert(0, "phase_mode", phase)
        table.to_csv(base.with_suffix(".csv"), index=False)
        points.to_csv(base.with_suffix(".points.csv"), index=False)
        plot(points, table, phase, edges, limits, label, base, reference=args.reference_kperp_rhoi)
        written.extend(str(base.with_suffix(s)) for s in EXTENSIONS)
    if written:
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
                    "selection": selection.selection_metadata(master, args.min_occupied_blocks),
                    "mlt_sector": "all", "valid_fraction_screening": False,
                    "x0_bin_width_dex": args.x0_bin_width_dex, "min_bin_count": args.min_bin_count,
                    "bin_edges_" + x_column: {c: e.tolist() for c, e in edges.items()},
                    "reference_kperp_rhoi": args.reference_kperp_rhoi,
                    "x_column": x_column,
                    "amplitude_formula": "log10_X_reference=log10_X0-kappa*log10(reference_kperp_rhoi)",
                    "amplitude_source": "saved fitted power law; not a newly measured spectrum",
                    "y_limits": limits,
                    "grid_config": styling.GRID_CONFIG, "plot_config": PLOT_CONFIG,
                    "master_sha256": provenance["annual_master_sha256"],
                    "source_provenance_sha256": batch.file_sha256(root / "inputs/provenance.json"),
                    "summary": "equal range weights; q16-q84 distribution spread, not confidence interval",
                    "correlation": "descriptive Spearman rho and Pearson r in plotted log10 amplitude versus kappa; shared-fit covariance not removed",
                    "code_sha256": {str(Path(p).resolve()): batch.file_sha256(Path(p)) for p in
                                    (__file__, ae.__file__, phases.__file__, selection.__file__, styling.__file__, correlations.__file__)},
                    "written": written}
        (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(written)} files: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
