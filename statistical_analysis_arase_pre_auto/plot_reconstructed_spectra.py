#!/usr/bin/env python3
"""Reconstruct range-fitted spectra versus k_perp*rho_i from saved fit pairs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_x0_kappa_relationship as relation
import statistics_phases as phases
import statistics_selection as selection
import statistics_plotting as styling


PLOT_CONFIG = {"dpi": 180, "font_size": 12}
COORDINATE = r"$k_\perp\rho_{\mathrm{i}}$"


def reconstruct(points, grid):
    """Equal range weights; quantiles computed in log10 amplitude (monotonic)."""
    tables, values = [], {}
    for component in relation.ae.COMPONENTS:
        group = points.loc[points.component.eq(component)]
        if group.empty:
            continue
        x = group.log10_X0.to_numpy(float)
        kappa = group.kappa.to_numpy(float)
        log_amplitude = x[:, None] - kappa[:, None] * np.log10(grid)[None, :]
        q16, median, q84 = np.quantile(log_amplitude, [.16, .5, .84], axis=0)
        std = np.std(log_amplitude, axis=0, ddof=1) if len(x) > 1 else np.full(len(grid), np.nan)
        width = q84 - q16
        low, high = np.quantile(x, [.25, .75])
        low_curve = np.median(log_amplitude[x <= low], axis=0)
        high_curve = np.median(log_amplitude[x >= high], axis=0)
        table = pd.DataFrame({"component": component, "kperp_rhoi": grid, "n": len(x),
                              "log10_X_q16": q16, "log10_X_median": median, "log10_X_q84": q84,
                              "log10_X_std": std, "log10_X_q84_minus_q16": width,
                              "low_X0_quartile_log10_X_median": low_curve,
                              "high_X0_quartile_log10_X_median": high_curve})
        tables.append(table)
        values[component] = log_amplitude
    return pd.concat(tables, ignore_index=True), values


def plot(points, table, values, title, base):
    with plt.rc_context({"font.size": PLOT_CONFIG["font_size"]}):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex="col")
        for col, component in enumerate(relation.ae.COMPONENTS):
            group = table.loc[table.component.eq(component)]
            if group.empty:
                continue
            x = group.kperp_rhoi.to_numpy()
            ax = axes[0, col]
            ax.plot(x, values[component].T, color=".5", alpha=.055, lw=.5, rasterized=True)
            ax.fill_between(x, group.log10_X_q16, group.log10_X_q84, color=".5", alpha=.2,
                            label="q16-q84 distribution spread")
            ax.plot(x, group.log10_X_median, color="black", lw=2, label="all-range median")
            ax.plot(x, group.low_X0_quartile_log10_X_median, color="tab:blue", lw=2, ls="--",
                    label="lowest X0 quartile median")
            ax.plot(x, group.high_X0_quartile_log10_X_median, color="tab:orange", lw=2, ls="--",
                    label="highest X0 quartile median")
            ax.set_title(f"{component}; N={int(group.n.iloc[0])}")
            ax.set_ylabel(rf"$\log_{{10}} X_{{\mathrm{{{component}}}}}(k_\perp\rho_{{\mathrm{{i}}}})$")
            ax.legend(fontsize=8)
            ax = axes[1, col]
            ax.plot(x, group.log10_X_std, color="tab:purple", label="std(log10 X)")
            ax.plot(x, group.log10_X_q84_minus_q16, color="tab:green", label="q84-q16(log10 X)")
            ax.set_ylabel("Range-to-range spread [dex]")
            ax.legend(fontsize=8)
            for row in range(2):
                ax = axes[row, col]
                ax.set_xscale("log")
                ax.set_xlim(x[0], x[-1])
                ax.set_xlabel(COORDINATE)
                if x[0] < 3:
                    ax.axvspan(x[0], min(3., x[-1]), color=".8", alpha=.3)
                if x[0] <= 3 <= x[-1]:
                    ax.axvline(3, color=".3", ls=":", lw=1)
                styling.detail_grid(ax)
        fig.suptitle(title + "\nReconstructed power laws; individual fit-domain coverage not applied"
                     + "\nShaded left region: below fit lower bound " + COORDINATE + " = 3")
        fig.tight_layout(rect=(0, 0, 1, .89))
        for suffix in (".png", ".pdf"):
            fig.savefig(base.with_suffix(suffix), dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    relation.add_period_arguments(parser)
    selection.add_selection_argument(parser)
    styling.add_output_arguments(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Content directory (reconstructed_spectra)")
    parser.add_argument("--phase", choices=tuple(phases.PHASES), action="append")
    parser.add_argument("--kperp-rhoi-min", type=float, default=1.)
    parser.add_argument("--kperp-rhoi-max", type=float, default=100.)
    parser.add_argument("--kperp-rhoi-points", type=int, default=201)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (not np.isfinite(args.kperp_rhoi_min) or not np.isfinite(args.kperp_rhoi_max)
            or not 0 < args.kperp_rhoi_min < args.kperp_rhoi_max or args.kperp_rhoi_points < 2):
        raise ValueError("Require 0 < kperp-rhoi-min < max, finite bounds and at least 2 points")
    grid = np.geomspace(args.kperp_rhoi_min, args.kperp_rhoi_max, args.kperp_rhoi_points)
    period = relation.resolve_period(args)
    root = args.statistics_dir or relation.batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks) / "reconstructed_spectra"
    master, provenance = selection.load_prepared_inputs(root, period)
    master = phases.classify_phases(master)
    selected = selection.select_events(master, args.min_occupied_blocks)
    output.mkdir(parents=True, exist_ok=True)
    for phase in dict.fromkeys(args.phase or phases.PHASES):
        base = output / f"arase_reconstructed_spectra_{phase}_{period.date_suffix}"
        if not styling.should_write(base, overwrite=not args.skip_existing,
                                    extensions=(".png", ".pdf", ".csv", ".points.csv", ".provenance.json")):
            continue
        points = relation.paired_points(selected, phase)
        if points.empty:
            continue
        table, values = reconstruct(points, grid)
        table.to_csv(base.with_suffix(".csv"), index=False)
        points.to_csv(base.with_suffix(".points.csv"), index=False)
        title = (f"Arase reconstructed spectra: {phases.PHASES[phase]}; {period.key}; all MLT; "
                 f"occupied 10 s blocks >= {args.min_occupied_blocks}")
        plot(points, table, values, title, base)
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "period": period.key, "phase": phase,
                    "formula": "log10_X(kperp_rhoi)=log10_X0-kappa*log10(kperp_rhoi)",
                    "grid": grid.tolist(), "grid_config": styling.GRID_CONFIG,
                    "selection": selection.selection_metadata(master, args.min_occupied_blocks),
                    "master_sha256": provenance["annual_master_sha256"],
                    "source_provenance_sha256": relation.batch.file_sha256(root / "inputs/provenance.json"),
                    "fit_domain_coverage_applied": False, "valid_fraction_screening": False,
                    "summary": "equal range weights; q16-q84 spread, not confidence interval; fixed X0 quartile groups",
                    "caveat": "model reconstructions, not measured spectra; ranges may extrapolate above their fit upper bound",
                    "code_sha256": {str(Path(p).resolve()): relation.batch.file_sha256(Path(p)) for p in
                                    (__file__, relation.__file__, phases.__file__, selection.__file__, styling.__file__)}}
        base.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote spectra: {base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
