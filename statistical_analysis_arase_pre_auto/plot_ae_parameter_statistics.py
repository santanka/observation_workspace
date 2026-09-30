#!/usr/bin/env python3
"""Plot six range-based KAW parameters against the AE index."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import batch_arase as batch
import plot_occurrence_probability as occurrence


PLOT_CONFIG = {
    "ae_bin_width_nT": 200.0,
    "ae_min_nT": 0.0,
    "ae_max_nT": None,
    "min_bin_count": 1,
    "ae_min_valid_fraction": 0.8,
    # None uses all MLT. Set (18.0, 6.0) for the nightside sensitivity test.
    "mlt_sector_hour": None,
    "show_ae_range_spread": False,
    "annotate_bin_count": True,
    "raw_marker_size": 22.0,
    "raw_alpha": 0.28,
    "component_colors": {"E": "tab:red", "B": "tab:blue", "S": "tab:green"},
    "figure_size": (17.0, 10.5),
    "dpi": 300,
    "font_size": 14,
}

PHASES = occurrence.PHASES
COMPONENTS = ("E", "B", "S")
FAMILIES = {
    "log10_x0": {
        "columns": {component: f"log10_X0_{component}" for component in COMPONENTS},
        "label": lambda component: rf"$\log_{{10}}(X_{{0,\mathrm{{{component}}}}})$",
    },
    "kappa": {
        "columns": {component: f"kappa_{component}" for component in COMPONENTS},
        "label": lambda component: rf"$\kappa_{{\mathrm{{{component}}}}}$",
    },
}
AE_COLUMN = "range_geomag_AE_nT_median"
AE_VALID_FRACTION_COLUMN = "range_geomag_AE_nT_valid_fraction"
AE_Q16_COLUMN = "range_geomag_AE_nT_q16"
AE_Q84_COLUMN = "range_geomag_AE_nT_q84"
SUMMARY_FIELDS = ("mean", "std", "median", "q16", "q84", "min", "max")


def resolve_paths(args, ranges_hash):
    key = batch.dataset_key(args.ranges, ranges_hash)
    state_dir = args.state_dir or batch.DEFAULT_STATE_ROOT / key
    master = args.master or state_dir / "aggregate" / "arase_phase_master.csv"
    output_dir = args.output_dir or batch.DEFAULT_OUTPUT_ROOT / key / "ae_statistics"
    return key, master, output_dir


def _summary(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    result = {"n": int(finite.size)}
    if finite.size == 0:
        result.update({name: np.nan for name in SUMMARY_FIELDS})
        return result
    q16, median, q84 = np.percentile(finite, [16.0, 50.0, 84.0])
    result.update({
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
        "median": float(median),
        "q16": float(q16),
        "q84": float(q84),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    })
    return result


def select_phase(master, phase_mode):
    required = {
        "phase_mode", "status", AE_COLUMN, AE_VALID_FRACTION_COLUMN,
        "range_MLT_circular_mean_hour", AE_Q16_COLUMN, AE_Q84_COLUMN,
    }
    required.update(
        column
        for family in FAMILIES.values()
        for column in family["columns"].values()
    )
    missing = required - set(master.columns)
    if missing:
        raise KeyError(f"Master CSV lacks columns: {sorted(missing)}")
    selected = master.loc[
        master["phase_mode"].eq(phase_mode) & master["status"].eq("ok")
    ].copy()
    ae = pd.to_numeric(selected[AE_COLUMN], errors="coerce")
    valid_fraction = pd.to_numeric(
        selected[AE_VALID_FRACTION_COLUMN], errors="coerce"
    )
    selected = selected.loc[
        ae.ge(PLOT_CONFIG["ae_min_nT"])
        & valid_fraction.ge(PLOT_CONFIG["ae_min_valid_fraction"])
    ].copy()
    sector = PLOT_CONFIG["mlt_sector_hour"]
    if sector is not None:
        selected = selected.loc[occurrence._in_wrapped_mlt_sector(
            selected["range_MLT_circular_mean_hour"], sector=sector
        )].copy()
    if selected.empty:
        raise ValueError(f"No valid AE rows for phase_mode={phase_mode}")
    return selected


def make_ae_edges(frames):
    width = float(PLOT_CONFIG["ae_bin_width_nT"])
    minimum = float(PLOT_CONFIG["ae_min_nT"])
    if not np.isfinite(width) or width <= 0:
        raise ValueError("ae_bin_width_nT must be positive")
    configured_maximum = PLOT_CONFIG["ae_max_nT"]
    if configured_maximum is None:
        values = pd.concat([
            pd.to_numeric(frame[AE_COLUMN], errors="coerce") for frame in frames
        ], ignore_index=True).to_numpy(float)
        values = values[np.isfinite(values) & (values >= minimum)]
        if values.size == 0:
            raise ValueError("No finite AE values for bin construction")
        maximum = minimum + (
            np.floor((float(np.max(values)) - minimum) / width) + 1.0
        ) * width
    else:
        maximum = float(configured_maximum)
    if not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError("ae_max_nT must exceed ae_min_nT")
    n_bin = int(np.ceil((maximum - minimum) / width))
    return minimum + np.arange(n_bin + 1, dtype=float) * width


def aggregate_ae_statistics(frame, phase_mode, ae_edges):
    ae = pd.to_numeric(frame[AE_COLUMN], errors="coerce").to_numpy(float)
    ae_bin = np.searchsorted(ae_edges, ae, side="right") - 1
    in_domain = (
        np.isfinite(ae) & (ae >= ae_edges[0]) & (ae < ae_edges[-1])
    )
    rows = []
    for family_name, family in FAMILIES.items():
        for component, source_column in family["columns"].items():
            parameter = pd.to_numeric(
                frame[source_column], errors="coerce"
            ).to_numpy(float)
            pair = in_domain & np.isfinite(parameter)
            if np.count_nonzero(pair) >= 2:
                correlation = spearmanr(ae[pair], parameter[pair])
                rho = float(correlation.statistic)
                p_value = float(correlation.pvalue)
            else:
                rho = p_value = np.nan
            overall = _summary(parameter[pair])
            for index in range(len(ae_edges) - 1):
                stats = _summary(parameter[pair & (ae_bin == index)])
                rows.append({
                    "phase_mode": phase_mode,
                    "family": family_name,
                    "component": component,
                    "source_column": source_column,
                    "ae_bin": index,
                    "ae_lower_nT": ae_edges[index],
                    "ae_upper_nT": ae_edges[index + 1],
                    "ae_center_nT": 0.5 * (ae_edges[index] + ae_edges[index + 1]),
                    **stats,
                    "overall_n": overall["n"],
                    "overall_median": overall["median"],
                    "overall_q16": overall["q16"],
                    "overall_q84": overall["q84"],
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                })
    return pd.DataFrame(rows)


def determine_y_limits(frames):
    result = {}
    for family_name, family in FAMILIES.items():
        result[family_name] = {}
        for component, source_column in family["columns"].items():
            values = pd.concat([
                pd.to_numeric(frame[source_column], errors="coerce")
                for frame in frames
            ], ignore_index=True).to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                result[family_name][component] = (0.0, 1.0)
                continue
            low, high = float(np.min(values)), float(np.max(values))
            pad = max(0.06 * (high - low), 0.05 if np.isclose(low, high) else 0.0)
            result[family_name][component] = (low - pad, high + pad)
    return result


def _panel_title(table, family_name, component, panel_label):
    row = table.loc[
        table["family"].eq(family_name) & table["component"].eq(component)
    ].iloc[0]
    label = FAMILIES[family_name]["label"](component)
    return (
        f"({panel_label}) {label}\n"
        f"median = {row['overall_median']:.3g} "
        f"[{row['overall_q16']:.3g}, {row['overall_q84']:.3g}], "
        f"N = {int(row['overall_n'])}\n"
        rf"$\rho_{{\mathrm{{S}}}}$ = {row['spearman_rho']:.2f}"
    )


def plot_ae_statistics(
    frame, table, phase_mode, ae_edges, y_limits, date_label, output_base,
    show=False,
):
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    fig, axes = plt.subplots(
        2, 3, figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"],
        sharex=True,
    )
    panel_index = 0
    legend_handles = None
    for row_index, family_name in enumerate(("log10_x0", "kappa")):
        family = FAMILIES[family_name]
        for column_index, component in enumerate(COMPONENTS):
            ax = axes[row_index, column_index]
            color = PLOT_CONFIG["component_colors"][component]
            ae = pd.to_numeric(frame[AE_COLUMN], errors="coerce").to_numpy(float)
            parameter = pd.to_numeric(
                frame[family["columns"][component]], errors="coerce"
            ).to_numpy(float)
            finite = np.isfinite(ae) & np.isfinite(parameter)
            if PLOT_CONFIG["show_ae_range_spread"]:
                q16 = pd.to_numeric(frame[AE_Q16_COLUMN], errors="coerce").to_numpy(float)
                q84 = pd.to_numeric(frame[AE_Q84_COLUMN], errors="coerce").to_numpy(float)
                xerr = np.vstack((ae - q16, q84 - ae))
                raw = ax.errorbar(
                    ae[finite], parameter[finite], xerr=xerr[:, finite], fmt="o",
                    markersize=np.sqrt(PLOT_CONFIG["raw_marker_size"]),
                    color="0.35", alpha=PLOT_CONFIG["raw_alpha"],
                    linewidth=0.6, label="range",
                )
            else:
                raw = ax.scatter(
                    ae[finite], parameter[finite],
                    s=PLOT_CONFIG["raw_marker_size"], color="0.35",
                    alpha=PLOT_CONFIG["raw_alpha"], edgecolors="none",
                    label="range",
                )
            rows = table.loc[
                table["family"].eq(family_name)
                & table["component"].eq(component)
                & table["n"].ge(int(PLOT_CONFIG["min_bin_count"]))
            ]
            center = rows["ae_center_nT"].to_numpy(float)
            median = rows["median"].to_numpy(float)
            yerr = np.vstack((
                median - rows["q16"].to_numpy(float),
                rows["q84"].to_numpy(float) - median,
            ))
            binned = ax.errorbar(
                center, median, yerr=yerr, fmt="o-", color=color,
                linewidth=1.8, markersize=6.0, capsize=3.0,
                label="bin median [q16, q84]", zorder=4,
            )
            if PLOT_CONFIG["annotate_bin_count"]:
                for x_value, y_value, count in zip(center, median, rows["n"]):
                    ax.annotate(
                        f"n={int(count)}", (x_value, y_value), xytext=(0, 8),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=color,
                    )
            panel_label = chr(ord("a") + panel_index)
            ax.set_title(
                _panel_title(table, family_name, component, panel_label),
                pad=12, fontsize=13,
            )
            ax.set_xlim(ae_edges[0], ae_edges[-1])
            ax.set_ylim(*y_limits[family_name][component])
            ax.grid(True, color="0.75", alpha=0.45, linewidth=0.7)
            if row_index == 1:
                ax.set_xlabel("Range median AE [nT]")
            ax.set_ylabel(family["label"](component))
            if legend_handles is None:
                legend_handles = (raw, binned)
            panel_index += 1

    sector = PLOT_CONFIG["mlt_sector_hour"]
    sector_label = "all MLT" if sector is None else f"{sector[0]:g}-{sector[1]:g} MLT"
    fig.suptitle(
        f"Arase KAW parameters versus AE: {PHASES[phase_mode]} ({sector_label})\n"
        f"{date_label}"
    )
    if legend_handles is not None:
        fig.legend(
            legend_handles, ("range", "bin median [q16, q84]"),
            loc="lower center", ncol=2, frameon=False,
        )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, default=batch.DEFAULT_RANGES)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--phase", action="append", choices=tuple(PHASES),
        help="Phase to plot; repeat for multiple phases. Default: all four.",
    )
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    payload, ranges = batch.load_ranges(args.ranges)
    ranges_hash = batch.file_sha256(args.ranges)
    key, master_path, output_dir = resolve_paths(args, ranges_hash)
    if not master_path.is_file():
        raise FileNotFoundError(f"Master CSV not found: {master_path}")
    start, end = occurrence.manifest_time_window(payload, ranges)
    last_day = end - pd.Timedelta(1, unit="ns")
    date_label = f"{start:%Y-%m-%d} to {last_day:%Y-%m-%d}"
    master = pd.read_csv(master_path)
    phases = list(dict.fromkeys(args.phase or PHASES))
    frames = {phase: select_phase(master, phase) for phase in phases}
    ae_edges = make_ae_edges(list(frames.values()))
    y_limits = determine_y_limits(list(frames.values()))

    written = []
    for phase_mode, frame in frames.items():
        table = aggregate_ae_statistics(frame, phase_mode, ae_edges)
        stem = (
            f"arase_ae_parameter_statistics_{phase_mode}_"
            f"{start:%Y%m%d}_{last_day:%Y%m%d}"
        )
        csv_path = output_dir / f"{stem}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        written.append(csv_path)
        written.extend(plot_ae_statistics(
            frame, table, phase_mode, ae_edges, y_limits, date_label,
            output_dir / stem, show=args.show,
        ))
        print(
            f"{phase_mode}: ranges={len(frame)}, "
            f"AE={frame[AE_COLUMN].min():.1f}-{frame[AE_COLUMN].max():.1f} nT"
        )

    print(f"dataset: {key}")
    print(f"master: {master_path}")
    print(f"AE edges [nT]: {ae_edges.tolist()}")
    print(f"min_bin_count: {PLOT_CONFIG['min_bin_count']}")
    for path in written:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
