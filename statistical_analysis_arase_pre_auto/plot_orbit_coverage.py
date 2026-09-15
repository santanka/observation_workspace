#!/usr/bin/env python3
"""Plot Arase GSM positions restricted to intervals in a valid-ranges JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import batch_arase as batch

os.environ.setdefault("MPLCONFIGDIR", str(batch.ROOT / ".mplconfig"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Plot settings intended for manual adjustment.
# Set a limit to None to determine it automatically from the plotted points.
# Limit pairs may be descending to reverse an axis, following the reference
# figure KAW_observation/KAW_observation_2025_fig1.ipynb.
# ---------------------------------------------------------------------------
PLOT_CONFIG = {
    # None uses the earliest/latest day represented in the ranges JSON.
    # Explicit ISO timestamps may still be supplied to plot a subset.
    "time_start": None,
    "time_end": None,  # exclusive
    "cmap": "turbo",
    "marker_size": 4.0,
    "marker_alpha": 0.85,
    "figure_size": (18.0, 8),
    "dpi": 300,
    "font_size": 20,
    "earth_dayside_color": "white",
    "earth_nightside_color": "black",
    "earth_edgecolor": "black",
    "grid_alpha": 0.45,
    "axis_limits": {
        "xy": {"x": None, "y": None},
        "xz": {"x": None, "y": None},
        "yz": {"x": None, "y": None},
    },
    "reverse_axes": {
        "xy_x": True,
        "xy_y": True,
        "xz_x": True,
        "xz_y": False,
        "yz_x": True,
        "yz_y": False,
    },
    # None chooses 5--8 readable ticks for any year and duration.
    "colorbar_ticks": None,
    "colorbar_format": None,
    # [left, bottom, width, height] in figure coordinates.
    "colorbar_axes": [0.29, 0.15, 0.42, 0.018],
}

PHASES = {
    "north_traveling": "ranges with northward-traveling detections",
    "south_traveling": "ranges with southward-traveling detections",
    "standing": "ranges with standing-wave-like detections",
    "all": "ranges with detections in the all-phase analysis",
}


def clipped_ranges(ranges, start: pd.Timestamp, end: pd.Timestamp):
    selected = []
    for item in ranges:
        left = max(pd.Timestamp(item["start_time"]), start)
        right = min(pd.Timestamp(item["end_time"]), end)
        if left < right:
            selected.append({
                "range_id": int(item["range_id"]),
                "start": left,
                "end": right,
            })
    if not selected:
        raise ValueError(f"No valid ranges overlap {start} -- {end}")
    return selected


def resolve_plot_window(ranges):
    starts = pd.to_datetime([item["start_time"] for item in ranges])
    ends = pd.to_datetime([item["end_time"] for item in ranges])
    configured_start = PLOT_CONFIG["time_start"]
    configured_end = PLOT_CONFIG["time_end"]
    start = (
        pd.Timestamp(configured_start)
        if configured_start is not None else pd.Timestamp(starts.min()).floor("D")
    )
    end = (
        pd.Timestamp(configured_end)
        if configured_end is not None else pd.Timestamp(ends.max()).ceil("D")
    )
    if start >= end:
        raise ValueError(f"Plot time range must increase: {start} -- {end}")
    return start, end


def ranges_by_day(ranges):
    result: dict[pd.Timestamp, list[dict]] = {}
    one_ns = pd.Timedelta(1, unit="ns")
    for item in ranges:
        last_inclusive = item["end"] - one_ns
        for day in pd.date_range(item["start"].normalize(), last_inclusive.normalize()):
            day_start = pd.Timestamp(day)
            day_end = day_start + pd.Timedelta(days=1)
            result.setdefault(day_start, []).append({
                "range_id": item["range_id"],
                "start": max(item["start"], day_start),
                "end": min(item["end"], day_end),
            })
    return result


def load_orbit_points(ranges, spedas_dir: Path) -> pd.DataFrame:
    os.environ["SPEDAS_DATA_DIR"] = str(spedas_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(batch.ROOT / ".mplconfig"))
    import ergpyspedas.erg as ergpy
    import pyspedas as psp
    import pytplot as pt

    frames = []
    missing_days = []
    for day, intervals in sorted(ranges_by_day(ranges).items()):
        day_end = day + pd.Timedelta(days=1)
        pt.del_data("erg_orb_l2_*")
        loaded = ergpy.orb(
            trange=[day.isoformat(), day_end.isoformat()],
            level="l2", datatype="def", no_update=True,
        )
        da = psp.get_data("erg_orb_l2_pos_gsm", xarray=True)
        if not loaded or da is None or da.time.size == 0:
            missing_days.append(day.strftime("%Y-%m-%d"))
            continue
        units = str(da.attrs.get("data_att", {}).get("units", "")).upper()
        if units not in {"RE", "R_E"}:
            raise ValueError(f"Unexpected orbit position unit on {day.date()}: {units!r}")
        time = pd.DatetimeIndex(pd.to_datetime(da.time.values))
        xyz = np.asarray(da.values, dtype=float)
        for interval in intervals:
            mask = (time >= interval["start"]) & (time <= interval["end"])
            if not np.any(mask):
                continue
            frames.append(pd.DataFrame({
                "time": time[mask],
                "x_gsm_re": xyz[mask, 0],
                "y_gsm_re": xyz[mask, 1],
                "z_gsm_re": xyz[mask, 2],
                "range_id": interval["range_id"],
            }))
    if missing_days:
        raise RuntimeError(
            "Orbit data unavailable in local cache for: " + ", ".join(missing_days)
        )
    if not frames:
        raise RuntimeError("No orbit samples overlap the selected valid ranges")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time", "range_id"])
        .sort_values(["time", "range_id"])
        .reset_index(drop=True)
    )


def write_cache(path: Path, points: pd.DataFrame, ranges_hash: str, start, end):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ns=points["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        xyz_gsm_re=points[["x_gsm_re", "y_gsm_re", "z_gsm_re"]].to_numpy(float),
        range_id=points["range_id"].to_numpy(int),
        ranges_sha256=np.asarray(ranges_hash),
        time_start=np.asarray(str(start)),
        time_end=np.asarray(str(end)),
    )


def read_cache(path: Path, ranges_hash: str, start, end):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["ranges_sha256"]) != ranges_hash
            or str(data["time_start"]) != str(start)
            or str(data["time_end"]) != str(end)
        ):
            return None
        xyz = data["xyz_gsm_re"]
        return pd.DataFrame({
            "time": pd.to_datetime(data["time_ns"], unit="ns"),
            "x_gsm_re": xyz[:, 0],
            "y_gsm_re": xyz[:, 1],
            "z_gsm_re": xyz[:, 2],
            "range_id": data["range_id"].astype(int),
        })


def automatic_limits(a, b, reverse=False):
    values = np.concatenate([np.asarray(a, float), np.asarray(b, float), [-1.0, 1.0]])
    finite = values[np.isfinite(values)]
    low, high = float(np.min(finite)), float(np.max(finite))
    span = max(high - low, 1.0)
    limits = (low - 0.06 * span, high + 0.06 * span)
    return limits[::-1] if reverse else limits


def add_earth(ax, panel_key):
    if panel_key in {"xy", "xz"}:
        # GSM +X points sunward: +X is the white dayside and -X the black nightside.
        ax.add_patch(Wedge(
            (0, 0), 1.0, -90, 90,
            facecolor=PLOT_CONFIG["earth_dayside_color"],
            edgecolor="none", zorder=3,
        ))
        ax.add_patch(Wedge(
            (0, 0), 1.0, 90, 270,
            facecolor=PLOT_CONFIG["earth_nightside_color"],
            edgecolor="none", zorder=3,
        ))
    else:
        # X is perpendicular to the Y-Z projection; follow the reference figure.
        ax.add_patch(Circle(
            (0, 0), 1.0,
            facecolor=PLOT_CONFIG["earth_nightside_color"],
            edgecolor="none", zorder=3,
        ))
    ax.add_patch(Circle(
        (0, 0), 1.0, facecolor="none",
        edgecolor=PLOT_CONFIG["earth_edgecolor"], linewidth=1.0, zorder=4,
    ))


def style_panel(ax, xlabel, ylabel, x, y, panel_key):
    reverse = PLOT_CONFIG["reverse_axes"]
    configured = PLOT_CONFIG["axis_limits"][panel_key]
    xlim = configured["x"]
    ylim = configured["y"]
    if xlim is None:
        xlim = automatic_limits(x, [], reverse[f"{panel_key}_x"])
    if ylim is None:
        ylim = automatic_limits(y, [], reverse[f"{panel_key}_y"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    ax.grid(which="both", linestyle="--", alpha=PLOT_CONFIG["grid_alpha"])
    ax.axhline(0, color="black", lw=0.8, linestyle="-.", alpha=0.7)
    ax.axvline(0, color="black", lw=0.8, linestyle="-.", alpha=0.7)
    add_earth(ax, panel_key)


def successful_range_ids(master: pd.DataFrame, phase_mode: str) -> set[int]:
    required = {"range_id", "phase_mode", "status", "kappa_E", "kappa_B"}
    missing = required - set(master.columns)
    if missing:
        raise KeyError(f"Master CSV lacks columns: {sorted(missing)}")
    mask = (
        master["phase_mode"].eq(phase_mode)
        & master["status"].eq("ok")
        & np.isfinite(master["kappa_E"].to_numpy(dtype=float))
        & np.isfinite(master["kappa_B"].to_numpy(dtype=float))
    )
    return set(master.loc[mask, "range_id"].astype(int))


def plot_points(
    points, start, end, output_base: Path, show=False,
    figure_label="valid time ranges", limit_points=None,
):
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    dates = mdates.date2num(pd.to_datetime(points["time"]).dt.to_pydatetime())
    norm = Normalize(mdates.date2num(start.to_pydatetime()), mdates.date2num(end.to_pydatetime()))
    cmap = PLOT_CONFIG["cmap"]
    scatter_kwargs = {
        "c": dates, "cmap": cmap, "norm": norm,
        "s": PLOT_CONFIG["marker_size"],
        "alpha": PLOT_CONFIG["marker_alpha"],
        "edgecolors": "none", "rasterized": True,
    }
    fig, axes = plt.subplots(1, 3, figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"])
    x = points["x_gsm_re"].to_numpy()
    y = points["y_gsm_re"].to_numpy()
    z = points["z_gsm_re"].to_numpy()
    limits = points if limit_points is None else limit_points
    limit_x = limits["x_gsm_re"].to_numpy()
    limit_y = limits["y_gsm_re"].to_numpy()
    limit_z = limits["z_gsm_re"].to_numpy()
    scatters = [
        axes[0].scatter(x, y, **scatter_kwargs),
        axes[1].scatter(x, z, **scatter_kwargs),
        axes[2].scatter(y, z, **scatter_kwargs),
    ]
    style_panel(axes[0], r"$X_{\mathrm{GSM}}$ [$R_{\mathrm{E}}$]", r"$Y_{\mathrm{GSM}}$ [$R_{\mathrm{E}}$]", limit_x, limit_y, "xy")
    style_panel(axes[1], r"$X_{\mathrm{GSM}}$ [$R_{\mathrm{E}}$]", r"$Z_{\mathrm{GSM}}$ [$R_{\mathrm{E}}$]", limit_x, limit_z, "xz")
    style_panel(axes[2], r"$Y_{\mathrm{GSM}}$ [$R_{\mathrm{E}}$]", r"$Z_{\mathrm{GSM}}$ [$R_{\mathrm{E}}$]", limit_y, limit_z, "yz")
    for ax, title in zip(axes, ("(a) X–Y", "(b) X–Z", "(c) Y–Z")):
        ax.set_title(title)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.29, wspace=0.28)
    colorbar_ax = fig.add_axes(PLOT_CONFIG["colorbar_axes"])
    colorbar = fig.colorbar(scatters[0], cax=colorbar_ax, orientation="horizontal")
    configured_ticks = PLOT_CONFIG["colorbar_ticks"]
    configured_format = PLOT_CONFIG["colorbar_format"]
    if configured_ticks is None:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
        colorbar.ax.xaxis.set_major_locator(locator)
        colorbar.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ticks = pd.to_datetime(configured_ticks)
        colorbar.set_ticks(mdates.date2num(ticks.to_pydatetime()))
        if configured_format is not None:
            colorbar.ax.xaxis.set_major_formatter(mdates.DateFormatter(configured_format))
    colorbar.set_label("UTC date")
    fig.suptitle(
        f"Arase positions during {figure_label}\n"
        f"{start:%Y-%m-%d} to {(end - pd.Timedelta(1, unit='ns')):%Y-%m-%d} (GSM)",
        y=0.98,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, default=batch.DEFAULT_RANGES)
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--master", type=Path)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--detected-only", action="store_true",
        help="Write only separate orbit figures for successful ranges in each phase mode",
    )
    output_group.add_argument(
        "--all-ranges-only", action="store_true",
        help="Write only the orbit figure containing all candidate ranges",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    _, ranges = batch.load_ranges(args.ranges)
    ranges_hash = batch.file_sha256(args.ranges)
    key = batch.dataset_key(args.ranges, ranges_hash)
    output_dir = args.output_dir or batch.DEFAULT_OUTPUT_ROOT / key / "orbit_coverage"
    start, end = resolve_plot_window(ranges)
    selected = clipped_ranges(ranges, start, end)
    cache_path = output_dir / "arase_gsm_valid_range_positions.npz"
    points = None if args.refresh_cache else read_cache(cache_path, ranges_hash, start, end)
    if points is None:
        points = load_orbit_points(selected, args.spedas_dir)
        write_cache(cache_path, points, ranges_hash, start, end)
        print(f"Wrote orbit cache: {cache_path}")
    else:
        print(f"Reused orbit cache: {cache_path}")
    last_day = end - pd.Timedelta(1, unit="ns")
    print(f"Ranges: {len(selected)}; orbit samples: {len(points)}")
    write_all_ranges = not args.detected_only
    write_detected_phases = not args.all_ranges_only

    if write_all_ranges:
        output_base = output_dir / (
            f"arase_gsm_valid_ranges_{start:%Y%m%d}_{last_day:%Y%m%d}"
        )
        for path in plot_points(points, start, end, output_base, show=args.show):
            print(f"Wrote: {path}")

    if write_detected_phases:
        state_dir = args.state_dir or batch.DEFAULT_STATE_ROOT / key
        master_path = args.master or state_dir / "aggregate" / "arase_phase_master.csv"
        if not master_path.is_file():
            raise FileNotFoundError(f"Master CSV not found: {master_path}")
        master = pd.read_csv(master_path)
        detected_dir = output_dir / "detected_phase"
        for phase_mode, figure_label in PHASES.items():
            range_ids = successful_range_ids(master, phase_mode)
            phase_points = points.loc[points["range_id"].isin(range_ids)].copy()
            if phase_points.empty:
                print(f"No successful orbit points for {phase_mode}; skipped")
                continue
            output_base = detected_dir / (
                f"arase_gsm_detected_{phase_mode}_{start:%Y%m%d}_{last_day:%Y%m%d}"
            )
            written = plot_points(
                phase_points, start, end, output_base, show=args.show,
                figure_label=f"{figure_label} ($N={len(range_ids)}$)",
                limit_points=points,
            )
            print(
                f"{phase_mode}: {len(range_ids)} ranges; "
                f"{len(phase_points)} orbit samples"
            )
            for path in written:
                print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
