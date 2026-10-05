#!/usr/bin/env python3
"""Plot separate GSM and SM orbit figures for every analysis range."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import batch_arase as batch

os.environ.setdefault("MPLCONFIGDIR", str(batch.ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


PLOT_CONFIG = {
    "marker_interval_minutes": 2.0,
    "path_color": "tab:blue",
    "path_linewidth": 1.6,
    "path_alpha": 0.9,
    "start_color": "tab:green",
    "start_marker_size": 90.0,
    "interval_color": "tab:blue",
    "interval_marker_size": 22.0,
    "end_color": "tab:red",
    "end_marker_size": 150.0,
    "figure_size": (17.0, 11.0),
    "overview_only_figure_size": (17.0, 5.8),
    "dpi": 300,
    "font_size": 15,
    "grid_alpha": 0.45,
    "show_event_zoom": True,
    "event_zoom_min_span_re": 0.04,
    "event_zoom_padding_fraction": 0.18,
    "earth_dayside_color": "white",
    "earth_nightside_color": "black",
    "earth_edgecolor": "black",
    "axis_limits": {
        "xy": {"x": None, "y": None},
        "xz": {"x": None, "y": None},
        "yz": {"x": None, "y": None},
    },
    "reverse_axes": {
        "xy_x": True, "xy_y": True,
        "xz_x": True, "xz_y": False,
        "yz_x": True, "yz_y": False,
    },
    "formats": ("png", "pdf"),
}

COORDINATES = ("gsm", "sm")


def required_days(ranges):
    days = set()
    one_ns = pd.Timedelta(1, unit="ns")
    for item in ranges:
        start = pd.Timestamp(item["start_time"])
        end = pd.Timestamp(item["end_time"])
        days.update(pd.date_range(start.normalize(), (end - one_ns).normalize()))
    return sorted(pd.Timestamp(day) for day in days)


def load_native_orbit(ranges, spedas_dir: Path) -> pd.DataFrame:
    """Load each required ORB day once and return native GSM/SM positions."""
    os.environ["SPEDAS_DATA_DIR"] = str(spedas_dir)
    import ergpyspedas.erg as ergpy
    import pyspedas as psp
    import pytplot as pt

    frames = []
    missing = []
    for day in required_days(ranges):
        day_end = day + pd.Timedelta(days=1)
        pt.del_data("erg_orb_l2_*")
        loaded = ergpy.orb(
            trange=[day.isoformat(), day_end.isoformat()],
            level="l2", datatype="def", no_update=True,
        )
        gsm = psp.get_data("erg_orb_l2_pos_gsm", xarray=True)
        sm = psp.get_data("erg_orb_l2_pos_sm", xarray=True)
        if not loaded or gsm is None or sm is None:
            missing.append(day.strftime("%Y-%m-%d"))
            continue
        for name, data in (("GSM", gsm), ("SM", sm)):
            units = str(data.attrs.get("data_att", {}).get("units", "")).upper()
            if units not in {"RE", "R_E"}:
                raise ValueError(
                    f"Unexpected {name} position unit on {day.date()}: {units!r}"
                )
        gsm_time = pd.DatetimeIndex(pd.to_datetime(gsm.time.values))
        sm_time = pd.DatetimeIndex(pd.to_datetime(sm.time.values))
        common, gsm_index, sm_index = np.intersect1d(
            gsm_time.to_numpy(dtype="datetime64[ns]"),
            sm_time.to_numpy(dtype="datetime64[ns]"),
            assume_unique=False, return_indices=True,
        )
        if common.size == 0:
            missing.append(day.strftime("%Y-%m-%d"))
            continue
        gsm_xyz = np.asarray(gsm.values, dtype=float)[gsm_index]
        sm_xyz = np.asarray(sm.values, dtype=float)[sm_index]
        frames.append(pd.DataFrame({
            "time": pd.to_datetime(common),
            "x_gsm_re": gsm_xyz[:, 0], "y_gsm_re": gsm_xyz[:, 1],
            "z_gsm_re": gsm_xyz[:, 2],
            "x_sm_re": sm_xyz[:, 0], "y_sm_re": sm_xyz[:, 1],
            "z_sm_re": sm_xyz[:, 2],
        }))
    if missing:
        raise RuntimeError(
            "GSM/SM orbit data unavailable in local cache for: "
            + ", ".join(missing)
        )
    if not frames:
        raise RuntimeError("No native GSM/SM orbit data were loaded")
    return (
        pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
        .sort_values("time").reset_index(drop=True)
    )


def _interpolate_columns(native, target_times, columns):
    source_ns = native["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    target_ns = pd.DatetimeIndex(target_times).to_numpy(
        dtype="datetime64[ns]"
    ).astype(np.int64)
    if (
        source_ns.size < 2
        or target_ns.min() < source_ns[0]
        or target_ns.max() > source_ns[-1]
    ):
        raise ValueError("Orbit data do not bracket the requested range boundary")
    return np.column_stack([
        np.interp(target_ns, source_ns, native[column].to_numpy(float))
        for column in columns
    ])


def event_orbit_points(native, item):
    """Return one event with exact, interpolated start and end positions."""
    start = pd.Timestamp(item["start_time"])
    end = pd.Timestamp(item["end_time"])
    if start >= end:
        raise ValueError(f"Non-increasing range_id={item['range_id']}")
    native_time = native["time"].to_numpy(dtype="datetime64[ns]")
    start64 = start.to_datetime64()
    end64 = end.to_datetime64()
    left = max(int(np.searchsorted(native_time, start64, side="right")) - 1, 0)
    right = min(int(np.searchsorted(native_time, end64, side="left")) + 1, len(native))
    local = native.iloc[left:right].copy()
    columns = [f"{axis}_{coord}_re" for coord in COORDINATES for axis in "xyz"]
    boundary = _interpolate_columns(local, [start, end], columns)
    inside = local.loc[
        local["time"].gt(start) & local["time"].lt(end), ["time", *columns]
    ]
    endpoints = pd.DataFrame(boundary, columns=columns)
    endpoints.insert(0, "time", [start, end])
    result = (
        pd.concat([endpoints.iloc[[0]], inside, endpoints.iloc[[1]]], ignore_index=True)
        .drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    )
    result.insert(0, "range_id", int(item["range_id"]))
    return result


def build_event_orbit_cache(native, ranges):
    frames = []
    failures = []
    for item in ranges:
        try:
            frames.append(event_orbit_points(native, item))
        except Exception as exc:
            failures.append(f"range {item['range_id']}: {exc}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return pd.concat(frames, ignore_index=True)


def write_cache(path, points, ranges_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ns=points["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        range_id=points["range_id"].to_numpy(int),
        xyz_gsm_re=points[[f"{axis}_gsm_re" for axis in "xyz"]].to_numpy(float),
        xyz_sm_re=points[[f"{axis}_sm_re" for axis in "xyz"]].to_numpy(float),
        ranges_sha256=np.asarray(ranges_hash),
    )


def read_cache(path, ranges_hash):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["ranges_sha256"]) != ranges_hash:
            return None
        gsm = data["xyz_gsm_re"]
        sm = data["xyz_sm_re"]
        return pd.DataFrame({
            "range_id": data["range_id"].astype(int),
            "time": pd.to_datetime(data["time_ns"], unit="ns"),
            "x_gsm_re": gsm[:, 0], "y_gsm_re": gsm[:, 1],
            "z_gsm_re": gsm[:, 2],
            "x_sm_re": sm[:, 0], "y_sm_re": sm[:, 1], "z_sm_re": sm[:, 2],
        })


def marker_points(points, coordinate):
    start = points["time"].iloc[0]
    end = points["time"].iloc[-1]
    step = pd.Timedelta(minutes=float(PLOT_CONFIG["marker_interval_minutes"]))
    if step <= pd.Timedelta(0):
        raise ValueError("marker_interval_minutes must be positive")
    times = []
    current = start + step
    while current < end:
        times.append(current)
        current += step
    columns = [f"{axis}_{coordinate}_re" for axis in "xyz"]
    if not times:
        return pd.DataFrame(columns=["time", *columns])
    values = _interpolate_columns(points, times, columns)
    result = pd.DataFrame(values, columns=columns)
    result.insert(0, "time", times)
    return result


def _automatic_limits(a, b, reverse=False):
    values = np.concatenate([np.asarray(a, float), np.asarray(b, float), [-1.0, 1.0]])
    finite = values[np.isfinite(values)]
    low, high = float(np.min(finite)), float(np.max(finite))
    span = max(high - low, 1.0)
    result = (low - 0.08 * span, high + 0.08 * span)
    return result[::-1] if reverse else result


def _equal_span_limits(xlim, ylim):
    """Expand the shorter dimension while preserving each axis direction."""
    reverse_x = xlim[0] > xlim[1]
    reverse_y = ylim[0] > ylim[1]
    x_low, x_high = min(xlim), max(xlim)
    y_low, y_high = min(ylim), max(ylim)
    span = max(x_high - x_low, y_high - y_low)
    x_center = 0.5 * (x_low + x_high)
    y_center = 0.5 * (y_low + y_high)
    equal_x = (x_center - 0.5 * span, x_center + 0.5 * span)
    equal_y = (y_center - 0.5 * span, y_center + 0.5 * span)
    return (
        equal_x[::-1] if reverse_x else equal_x,
        equal_y[::-1] if reverse_y else equal_y,
    )


def _add_earth(ax, panel_key):
    if panel_key in {"xy", "xz"}:
        ax.add_patch(Wedge(
            (0, 0), 1.0, -90, 90,
            facecolor=PLOT_CONFIG["earth_dayside_color"], edgecolor="none", zorder=3,
        ))
        ax.add_patch(Wedge(
            (0, 0), 1.0, 90, 270,
            facecolor=PLOT_CONFIG["earth_nightside_color"], edgecolor="none", zorder=3,
        ))
    else:
        ax.add_patch(Circle(
            (0, 0), 1.0, facecolor=PLOT_CONFIG["earth_nightside_color"],
            edgecolor="none", zorder=3,
        ))
    ax.add_patch(Circle(
        (0, 0), 1.0, facecolor="none",
        edgecolor=PLOT_CONFIG["earth_edgecolor"], linewidth=1.0, zorder=4,
    ))


def _event_zoom_limits(x, y, reverse_x=False, reverse_y=False):
    """Return equal-span local limits so a short event remains visible."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size == 0 or finite_y.size == 0:
        raise ValueError("Event orbit contains no finite positions")
    span = max(
        float(np.ptp(finite_x)), float(np.ptp(finite_y)),
        float(PLOT_CONFIG["event_zoom_min_span_re"]),
    )
    half_width = 0.5 * span * (
        1.0 + 2.0 * float(PLOT_CONFIG["event_zoom_padding_fraction"])
    )
    x_center = 0.5 * (float(np.min(finite_x)) + float(np.max(finite_x)))
    y_center = 0.5 * (float(np.min(finite_y)) + float(np.max(finite_y)))
    xlim = (x_center - half_width, x_center + half_width)
    ylim = (y_center - half_width, y_center + half_width)
    return (
        xlim[::-1] if reverse_x else xlim,
        ylim[::-1] if reverse_y else ylim,
    )


def _draw_event_path(ax, x, y, marker_x, marker_y, marker_scale=1.0):
    ax.plot(
        x, y, color=PLOT_CONFIG["path_color"],
        linewidth=PLOT_CONFIG["path_linewidth"],
        alpha=PLOT_CONFIG["path_alpha"], zorder=5,
    )
    if len(marker_x):
        ax.scatter(
            marker_x, marker_y,
            s=PLOT_CONFIG["interval_marker_size"] * marker_scale, marker=".",
            color=PLOT_CONFIG["interval_color"], zorder=7,
        )
    ax.scatter(
        [x[0]], [y[0]],
        s=PLOT_CONFIG["start_marker_size"] * marker_scale, marker="o",
        color=PLOT_CONFIG["start_color"], edgecolors="black",
        linewidths=0.7, zorder=8,
    )
    ax.scatter(
        [x[-1]], [y[-1]],
        s=PLOT_CONFIG["end_marker_size"] * marker_scale, marker="*",
        color=PLOT_CONFIG["end_color"], edgecolors="black",
        linewidths=0.6, zorder=9,
    )


def _style_axis(ax, panel_key, x, y, coordinate):
    configured = PLOT_CONFIG["axis_limits"][panel_key]
    reverse = PLOT_CONFIG["reverse_axes"]
    xlim = configured["x"] or _automatic_limits(
        x, [], reverse[f"{panel_key}_x"]
    )
    ylim = configured["y"] or _automatic_limits(
        y, [], reverse[f"{panel_key}_y"]
    )
    xlim, ylim = _equal_span_limits(xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    left, right = panel_key
    ax.set_xlabel(
        rf"${left.upper()}_{{\mathrm{{{coordinate.upper()}}}}}$ [$R_{{\mathrm{{E}}}}$]"
    )
    ax.set_ylabel(
        rf"${right.upper()}_{{\mathrm{{{coordinate.upper()}}}}}$ [$R_{{\mathrm{{E}}}}$]"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    ax.grid(which="both", linestyle="--", alpha=PLOT_CONFIG["grid_alpha"])
    ax.axhline(0, color="black", lw=0.8, linestyle="-.", alpha=0.7)
    ax.axvline(0, color="black", lw=0.8, linestyle="-.", alpha=0.7)
    _add_earth(ax, panel_key)


def _set_coordinate_labels(ax, panel_key, coordinate):
    left, right = panel_key
    ax.set_xlabel(
        rf"${left.upper()}_{{\mathrm{{{coordinate.upper()}}}}}$ [$R_{{\mathrm{{E}}}}$]"
    )
    ax.set_ylabel(
        rf"${right.upper()}_{{\mathrm{{{coordinate.upper()}}}}}$ [$R_{{\mathrm{{E}}}}$]"
    )


def plot_event(points, item, coordinate, output_base, formats=None, show=False):
    if coordinate not in COORDINATES:
        raise ValueError(f"Unknown coordinate: {coordinate}")
    formats = tuple(formats or PLOT_CONFIG["formats"])
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    show_zoom = bool(PLOT_CONFIG["show_event_zoom"])
    if show_zoom:
        fig, axes_grid = plt.subplots(
            2, 3, figsize=PLOT_CONFIG["figure_size"], dpi=PLOT_CONFIG["dpi"]
        )
        overview_axes = axes_grid[0]
        zoom_axes = axes_grid[1]
    else:
        fig, overview_axes = plt.subplots(
            1, 3, figsize=PLOT_CONFIG["overview_only_figure_size"],
            dpi=PLOT_CONFIG["dpi"],
        )
        zoom_axes = [None] * 3
    columns = {axis: f"{axis}_{coordinate}_re" for axis in "xyz"}
    xyz = {axis: points[column].to_numpy(float) for axis, column in columns.items()}
    intermediate = marker_points(points, coordinate)
    panels = (("xy", "x", "y"), ("xz", "x", "z"), ("yz", "y", "z"))
    for ax, zoom, (panel_key, left, right) in zip(
        overview_axes, zoom_axes, panels
    ):
        marker_x = (
            intermediate[columns[left]].to_numpy(float)
            if not intermediate.empty else np.asarray([])
        )
        marker_y = (
            intermediate[columns[right]].to_numpy(float)
            if not intermediate.empty else np.asarray([])
        )
        _draw_event_path(
            ax, xyz[left], xyz[right], marker_x, marker_y,
        )
        _style_axis(ax, panel_key, xyz[left], xyz[right], coordinate)
        if zoom is not None:
            _draw_event_path(
                zoom, xyz[left], xyz[right], marker_x, marker_y,
                marker_scale=0.65,
            )
            reverse = PLOT_CONFIG["reverse_axes"]
            xlim, ylim = _event_zoom_limits(
                xyz[left], xyz[right],
                reverse[f"{panel_key}_x"], reverse[f"{panel_key}_y"],
            )
            zoom.set_xlim(*xlim)
            zoom.set_ylim(*ylim)
            zoom.set_aspect("equal", adjustable="box")
            zoom.grid(linestyle="--", alpha=PLOT_CONFIG["grid_alpha"])
            zoom.xaxis.set_major_locator(MaxNLocator(nbins=3))
            zoom.yaxis.set_major_locator(MaxNLocator(nbins=3))
            zoom.tick_params(labelsize=max(PLOT_CONFIG["font_size"] - 3, 8))
            _set_coordinate_labels(zoom, panel_key, coordinate)
    for ax, title in zip(
        overview_axes, ("(a) X–Y overview", "(b) X–Z overview", "(c) Y–Z overview")
    ):
        ax.set_title(title)
    if show_zoom:
        for ax, title in zip(
            zoom_axes,
            ("(d) X–Y event detail", "(e) X–Z event detail", "(f) Y–Z event detail"),
        ):
            ax.set_title(title)
    start = pd.Timestamp(item["start_time"])
    end = pd.Timestamp(item["end_time"])
    duration_min = (end - start).total_seconds() / 60.0
    fig.suptitle(
        f"Arase orbit: range {int(item['range_id']):03d} ({coordinate.upper()})\n"
        f"{start:%Y-%m-%d %H:%M:%S} to {end:%H:%M:%S} UTC "
        f"({duration_min:.2f} min)"
    )
    handles = [
        Line2D([], [], color=PLOT_CONFIG["path_color"],
               lw=PLOT_CONFIG["path_linewidth"], label="orbit"),
        Line2D([], [], marker="o", color="none",
               markerfacecolor=PLOT_CONFIG["start_color"],
               markeredgecolor="black", markersize=9, label="start"),
        Line2D([], [], marker=".", color="none",
               markerfacecolor=PLOT_CONFIG["interval_color"],
               markeredgecolor=PLOT_CONFIG["interval_color"], markersize=8,
               label=f"every {PLOT_CONFIG['marker_interval_minutes']:g} min"),
        Line2D([], [], marker="*", color="none",
               markerfacecolor=PLOT_CONFIG["end_color"],
               markeredgecolor="black", markersize=12, label="end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    if show_zoom:
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.86, bottom=0.10,
            hspace=0.55, wspace=0.27,
        )
    else:
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.78, bottom=0.20, wspace=0.27
        )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = output_base.with_suffix(f".{extension}")
        kwargs = {"dpi": PLOT_CONFIG["dpi"]} if extension.lower() == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    if show:
        plt.show()
    plt.close(fig)
    return paths


def completed_range_ids(state_dir: Path):
    path = state_dir / "aggregate" / "range_status.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Range status CSV not found: {path}")
    frame = pd.read_csv(path)
    return set(frame.loc[frame["status"].eq("complete"), "range_id"].astype(int))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranges", type=Path, default=batch.DEFAULT_RANGES)
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--range-id", type=int, action="append")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--png-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    _, all_ranges = batch.load_ranges(args.ranges)
    ranges_hash = batch.file_sha256(args.ranges)
    key = batch.dataset_key(args.ranges, ranges_hash)
    output_dir = args.output_dir or batch.DEFAULT_OUTPUT_ROOT / key / "event_orbits"
    state_dir = args.state_dir or batch.DEFAULT_STATE_ROOT / key
    cache_path = output_dir / "arase_gsm_sm_event_orbit_positions.npz"
    points = None if args.refresh_cache else read_cache(cache_path, ranges_hash)
    if points is None:
        native = load_native_orbit(all_ranges, args.spedas_dir)
        points = build_event_orbit_cache(native, all_ranges)
        write_cache(cache_path, points, ranges_hash)
        print(f"Wrote event-orbit cache: {cache_path}")
    else:
        print(f"Reused event-orbit cache: {cache_path}")

    selected = all_ranges
    if args.range_id:
        wanted = set(args.range_id)
        selected = [item for item in selected if int(item["range_id"]) in wanted]
        missing = wanted - {int(item["range_id"]) for item in selected}
        if missing:
            raise ValueError(f"Unknown range IDs: {sorted(missing)}")
    if args.completed_only:
        complete = completed_range_ids(state_dir)
        selected = [item for item in selected if int(item["range_id"]) in complete]
    if not selected:
        raise ValueError("No ranges selected for event-orbit plotting")

    formats = ("png",) if args.png_only else tuple(PLOT_CONFIG["formats"])
    written = 0
    skipped = 0
    for index, item in enumerate(selected, start=1):
        range_id = int(item["range_id"])
        event = points.loc[points["range_id"].eq(range_id)].sort_values("time")
        if event.empty:
            raise RuntimeError(f"No cached orbit points for range_id={range_id}")
        start = pd.Timestamp(item["start_time"])
        event_dir = output_dir / f"range_{range_id:03d}"
        for coordinate in COORDINATES:
            stem = (
                f"arase_orbit_{coordinate}_range_{range_id:03d}_"
                f"{start:%Y%m%d_%H%M%S}"
            )
            output_base = event_dir / stem
            expected = [
                output_base.with_suffix(f".{extension}") for extension in formats
            ]
            if not args.force and all(path.is_file() for path in expected):
                skipped += len(expected)
                continue
            written += len(plot_event(
                event, item, coordinate, output_base,
                formats=formats, show=args.show,
            ))
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(
                f"event-orbit progress: {index}/{len(selected)} ranges; "
                f"written={written}, skipped={skipped}"
            )
    print(f"dataset: {key}")
    print(f"ranges plotted: {len(selected)}")
    print(f"files written: {written}; existing files skipped: {skipped}")
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
