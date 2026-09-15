#!/usr/bin/env python3
"""Plot preliminary kappa_E and kappa_B time series for each phase mode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import batch_arase as batch

os.environ.setdefault("MPLCONFIGDIR", str(batch.ROOT / ".mplconfig"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Settings intended for later manual adjustment.
PLOT_CONFIG = {
    "kappa_E_color": "green",
    "kappa_B_color": "purple",
    "marker": "o",
    "marker_size": 4.5,
    "error_linewidth": 1.0,
    "capsize": 2.5,
    "alpha": 0.85,
    "mean_line_style": "--",
    "mean_line_width": 1.5,
    "mean_line_alpha": 0.9,
    "mean_decimals": 3,
    "figure_size": (12.0, 5.5),
    "dpi": 300,
    "font_size": 14,
    "grid_alpha": 0.45,
    # None gives a common automatic y range across all four phase figures.
    "ylim": None,
    # None uses the complete time extent of the ranges JSON.
    "time_start": None,
    "time_end": None,
}


PHASES = {
    "north_traveling": "Northward traveling",
    "south_traveling": "Southward traveling",
    "standing": "Standing-wave-like",
    "all": "All phase components",
}

REQUIRED_COLUMNS = {
    "range_start", "range_end", "phase_mode", "status",
    "kappa_E", "kappa_E_q16", "kappa_E_q84",
    "kappa_B", "kappa_B_q16", "kappa_B_q84",
}


def resolve_paths(args, ranges_hash):
    key = batch.dataset_key(args.ranges, ranges_hash)
    state_dir = args.state_dir or batch.DEFAULT_STATE_ROOT / key
    master = args.master or state_dir / "aggregate" / "arase_phase_master.csv"
    output_dir = args.output_dir or batch.DEFAULT_OUTPUT_ROOT / key / "kappa_time_series"
    return key, master, output_dir


def resolve_time_window(ranges):
    start = (
        pd.Timestamp(PLOT_CONFIG["time_start"])
        if PLOT_CONFIG["time_start"] is not None
        else pd.to_datetime([r["start_time"] for r in ranges]).min().floor("D")
    )
    end = (
        pd.Timestamp(PLOT_CONFIG["time_end"])
        if PLOT_CONFIG["time_end"] is not None
        else pd.to_datetime([r["end_time"] for r in ranges]).max().ceil("D")
    )
    if start >= end:
        raise ValueError(f"Plot time range must increase: {start} -- {end}")
    return start, end


def select_detected_results(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(f"Master CSV lacks columns: {sorted(missing)}")
    selected = frame.loc[frame["status"].eq("ok")].copy()
    numeric = [
        "kappa_E", "kappa_E_q16", "kappa_E_q84",
        "kappa_B", "kappa_B_q16", "kappa_B_q84",
    ]
    finite = np.isfinite(selected[numeric].to_numpy(dtype=float)).all(axis=1)
    ordered = (
        (selected["kappa_E_q16"] <= selected["kappa_E"])
        & (selected["kappa_E"] <= selected["kappa_E_q84"])
        & (selected["kappa_B_q16"] <= selected["kappa_B"])
        & (selected["kappa_B"] <= selected["kappa_B_q84"])
    )
    selected = selected.loc[finite & ordered].copy()
    selected["range_start"] = pd.to_datetime(selected["range_start"])
    selected["range_end"] = pd.to_datetime(selected["range_end"])
    selected["plot_time"] = selected["range_start"] + (
        selected["range_end"] - selected["range_start"]
    ) / 2
    return selected.sort_values(["plot_time", "phase_mode"]).reset_index(drop=True)


def common_ylim(frame: pd.DataFrame):
    if PLOT_CONFIG["ylim"] is not None:
        return tuple(PLOT_CONFIG["ylim"])
    low = float(frame[["kappa_E_q16", "kappa_B_q16"]].min().min())
    high = float(frame[["kappa_E_q84", "kappa_B_q84"]].max().max())
    span = max(high - low, 0.5)
    return low - 0.07 * span, high + 0.07 * span


def asymmetric_error(frame: pd.DataFrame, component: str):
    center = frame[f"kappa_{component}"].to_numpy(float)
    lower = center - frame[f"kappa_{component}_q16"].to_numpy(float)
    upper = frame[f"kappa_{component}_q84"].to_numpy(float) - center
    return np.vstack([lower, upper])


def phase_means(frame: pd.DataFrame):
    return float(frame["kappa_E"].mean()), float(frame["kappa_B"].mean())


def plot_phase(frame, phase_mode, title, start, end, ylim, output_base, show=False):
    phase = frame.loc[frame["phase_mode"].eq(phase_mode)].copy()
    if phase.empty:
        print(f"No detected results for {phase_mode}; skipped")
        return []
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"])
    shared = {
        "fmt": PLOT_CONFIG["marker"],
        "markersize": PLOT_CONFIG["marker_size"],
        "elinewidth": PLOT_CONFIG["error_linewidth"],
        "capsize": PLOT_CONFIG["capsize"],
        "alpha": PLOT_CONFIG["alpha"],
        "linestyle": "none",
    }
    ax.errorbar(
        phase["plot_time"], phase["kappa_E"],
        yerr=asymmetric_error(phase, "E"),
        color=PLOT_CONFIG["kappa_E_color"], ecolor=PLOT_CONFIG["kappa_E_color"],
        label=r"$\kappa_{\mathrm{E}}$", **shared,
    )
    ax.errorbar(
        phase["plot_time"], phase["kappa_B"],
        yerr=asymmetric_error(phase, "B"),
        color=PLOT_CONFIG["kappa_B_color"], ecolor=PLOT_CONFIG["kappa_B_color"],
        label=r"$\kappa_{\mathrm{B}}$", **shared,
    )
    mean_e, mean_b = phase_means(phase)
    mean_line = {
        "linestyle": PLOT_CONFIG["mean_line_style"],
        "linewidth": PLOT_CONFIG["mean_line_width"],
        "alpha": PLOT_CONFIG["mean_line_alpha"],
    }
    ax.axhline(mean_e, color=PLOT_CONFIG["kappa_E_color"], **mean_line)
    ax.axhline(mean_b, color=PLOT_CONFIG["kappa_B_color"], **mean_line)
    ax.set_xlim(start, end)
    ax.set_ylim(*ylim)
    ax.set_ylabel(r"Spectral index $\kappa$")
    ax.set_xlabel("UTC date")
    decimals = int(PLOT_CONFIG["mean_decimals"])
    ax.set_title(
        f"{title}  ($N={len(phase)}$ detected ranges)\n"
        rf"mean $\kappa_E={mean_e:.{decimals}f}$, "
        rf"mean $\kappa_B={mean_b:.{decimals}f}$"
    )
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.minorticks_on()
    ax.grid(which="both", linestyle="--", alpha=PLOT_CONFIG["grid_alpha"])
    ax.legend()
    fig.tight_layout()
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
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    _, ranges = batch.load_ranges(args.ranges)
    ranges_hash = batch.file_sha256(args.ranges)
    key, master_path, output_dir = resolve_paths(args, ranges_hash)
    if not master_path.is_file():
        raise FileNotFoundError(f"Master CSV not found: {master_path}")
    selected = select_detected_results(pd.read_csv(master_path))
    if selected.empty:
        raise RuntimeError("No detected kappa results are available")
    start, end = resolve_time_window(ranges)
    ylim = common_ylim(selected)
    last_day = end - pd.Timedelta(1, unit="ns")
    written = []
    for phase_mode, title in PHASES.items():
        output_base = output_dir / (
            f"kappa_E_B_{phase_mode}_{start:%Y%m%d}_{last_day:%Y%m%d}"
        )
        written.extend(plot_phase(
            selected, phase_mode, title, start, end, ylim, output_base, args.show
        ))
    print(f"dataset: {key}")
    print(f"master: {master_path}")
    print(f"detected phase-range rows: {len(selected)}")
    for path in written:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
