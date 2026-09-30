#!/usr/bin/env python3
"""Plot Arase-dwell-normalized KAW occurrence in L-MLAT and L-MLT."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import batch_arase as batch

os.environ.setdefault("MPLCONFIGDIR", str(batch.ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd


PLOT_CONFIG = {
    "l_edges": np.arange(3.0, 12.0, 1.0),
    "mlat_edges_deg": np.arange(-50.0, 55.0, 5.0),
    "mlt_edges_hour": np.arange(0.0, 25.0, 1.0),
    "l_mlat_mlt_sector_hour": (18.0, 6.0),
    "min_dwelling_duration_min": 0.0,
    "orbit_gap_threshold_sec": 30.0,
    "time_cmap": "viridis",
    "probability_cmap": "turbo",
    "probability_limits_percent": (0.0, 100.0),
    "color_scale": "log",
    "time_log_limits_hour": (None, None),
    "probability_log_limits_percent": (None, 100.0),
    "missing_color": "0.82",
    "earth_dayside_color": "white",
    "earth_nightside_color": "black",
    "grid_color": "white",
    "grid_major_alpha": 0.42,
    "grid_minor_alpha": 0.22,
    "grid_major_linewidth": 0.7,
    "grid_minor_linewidth": 0.35,
    "earth_edgecolor": "black",
    "figure_size_meridional": (17.0, 6.0),
    "figure_size_polar": (17.0, 6.2),
    "dpi": 300,
    "font_size": 15,
}

ORBIT_DOWNLOAD_RETRIES = 3
ORBIT_RETRY_DELAYS_SEC = (10, 30)

PHASES = {
    "all": "All phase components",
    "north_traveling": "Northward traveling",
    "south_traveling": "Southward traveling",
    "standing": "Standing",
}

POSITION_COLUMNS = {
    "l_mlat": {
        "coordinate_column": "range_MLAT_deg_mean",
        "dwelling_column": "mlat_deg",
        "coordinate_name": "mlat_deg",
        "edges_key": "mlat_edges_deg",
    },
    "l_mlt": {
        "coordinate_column": "range_MLT_circular_mean_hour",
        "dwelling_column": "mlt_hour",
        "coordinate_name": "mlt_hour",
        "edges_key": "mlt_edges_hour",
    },
}

REQUIRED_COLUMNS = {
    "range_id", "range_duration_sec", "phase_mode", "status",
    "range_L_90deg_mean", "range_MLAT_deg_mean",
    "range_MLT_circular_mean_hour",
}


def resolve_paths(args, ranges_hash):
    key = batch.dataset_key(args.ranges, ranges_hash)
    state_dir = args.state_dir or batch.DEFAULT_STATE_ROOT / key
    master = args.master or state_dir / "aggregate" / "arase_phase_master.csv"
    output_dir = (
        args.output_dir
        or batch.DEFAULT_OUTPUT_ROOT / key / "occurrence_probability"
    )
    return key, master, output_dir


def _parse_manifest_time(value) -> pd.Timestamp:
    text = str(value)
    for fmt in ("%Y%m%d/%H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            pass
    return pd.Timestamp(text)


def manifest_time_window(payload, ranges):
    configured = payload.get("time_range_input")
    if isinstance(configured, list) and len(configured) == 2:
        start, end = map(_parse_manifest_time, configured)
    else:
        start = pd.to_datetime([item["start_time"] for item in ranges]).min()
        end = pd.to_datetime([item["end_time"] for item in ranges]).max()
    if start >= end:
        raise ValueError(f"Non-increasing manifest time range: {start} -- {end}")
    return start, end


def orbit_day_intervals(start, end):
    intervals = []
    for day in pd.date_range(start.floor("D"), end.ceil("D"), inclusive="left"):
        left = max(pd.Timestamp(day), start)
        right = min(pd.Timestamp(day) + pd.Timedelta(days=1), end)
        if left < right:
            intervals.append((left, right))
    return intervals


def _local_orbit_files(ergpy, start, end):
    files = ergpy.orb(
        trange=[start.isoformat(), end.isoformat()],
        level="l2", datatype="def", downloadonly=True, no_update=True,
    )
    return [Path(path) for path in (files or []) if Path(path).is_file()]


def ensure_orbit_files(
    start, end, spedas_dir: Path, retries=ORBIT_DOWNLOAD_RETRIES,
    retry_delays=ORBIT_RETRY_DELAYS_SEC, allow_download=True,
):
    """Return local daily ORB files, downloading only missing days with retries."""
    if retries < 1:
        raise ValueError("Orbit download retries must be at least 1")
    os.environ["SPEDAS_DATA_DIR"] = str(spedas_dir)
    import ergpyspedas.erg as ergpy

    result = []
    unresolved = []
    for day_start, day_end in orbit_day_intervals(start, end):
        files = _local_orbit_files(ergpy, day_start, day_end)
        if files:
            result.extend(files)
            continue
        if not allow_download:
            unresolved.append(day_start.strftime("%Y-%m-%d"))
            continue
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                print(
                    f"ORB download {day_start:%Y-%m-%d}: "
                    f"attempt {attempt}/{retries}"
                )
                downloaded = ergpy.orb(
                    trange=[day_start.isoformat(), day_end.isoformat()],
                    level="l2", datatype="def", downloadonly=True,
                )
                files = [
                    Path(path) for path in (downloaded or [])
                    if Path(path).is_file()
                ]
                if not files:
                    files = _local_orbit_files(ergpy, day_start, day_end)
                if files:
                    result.extend(files)
                    break
                last_error = "loader returned no local file"
            except Exception as exc:  # network/remote failures vary by backend
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                print(
                    f"ORB download retry for {day_start:%Y-%m-%d} "
                    f"in {delay}s ({last_error})"
                )
                time.sleep(delay)
        else:
            unresolved.append(f"{day_start:%Y-%m-%d} ({last_error})")
    if unresolved:
        raise RuntimeError(
            "ORB L2 remains unavailable; dwelling statistics were not written: "
            + ", ".join(unresolved)
        )
    return sorted(set(result))


def load_orbit_dwelling_points(start, end, spedas_dir: Path) -> pd.DataFrame:
    """Load native ORB samples and assign time weights without bridging gaps."""
    os.environ["SPEDAS_DATA_DIR"] = str(spedas_dir)
    import ergpyspedas.erg as ergpy
    import pyspedas as psp
    import pytplot as pt

    frames = []
    for day_start, day_end in orbit_day_intervals(start, end):
        pt.del_data("erg_orb_l2_*")
        loaded = ergpy.orb(
            trange=[day_start.isoformat(), day_end.isoformat()],
            level="l2", datatype="def", no_update=True,
        )
        rmlatmlt = psp.get_data("erg_orb_l2_pos_rmlatmlt", xarray=True)
        lm = psp.get_data("erg_orb_l2_pos_Lm", xarray=True)
        if not loaded or rmlatmlt is None or lm is None:
            raise RuntimeError(f"Failed to load local ORB L2 on {day_start:%Y-%m-%d}")
        rmlatmlt = rmlatmlt.sortby("time")
        lm = lm.sortby("time")
        time_index = pd.DatetimeIndex(pd.to_datetime(rmlatmlt.time.values))
        lm_time = pd.DatetimeIndex(pd.to_datetime(lm.time.values))
        if not time_index.equals(lm_time):
            raise ValueError(f"ORB position times are misaligned on {day_start:%Y-%m-%d}")
        values = np.asarray(rmlatmlt.values, dtype=float)
        l_values = np.asarray(lm.values, dtype=float)
        mask = (time_index >= day_start) & (time_index < day_end)
        frames.append(pd.DataFrame({
            "time": time_index[mask],
            "r_re": values[mask, 0],
            "mlat_deg": values[mask, 1],
            "mlt_hour": np.mod(values[mask, 2], 24.0),
            # CDF CATDESC lists descending pitch angle: 90, 60, 30 deg.
            "l_90deg": l_values[mask, 0],
        }))
    points = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="time")
        .sort_values("time")
        .reset_index(drop=True)
    )
    if points.empty:
        raise RuntimeError("No ORB samples are available in the manifest time range")
    time_ns = points["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    end_ns = pd.Timestamp(end).to_datetime64().astype("datetime64[ns]").astype(np.int64)
    dt_sec = np.diff(time_ns, append=end_ns) / 1e9
    threshold = float(PLOT_CONFIG["orbit_gap_threshold_sec"])
    points["dwell_sec"] = np.where(
        (dt_sec > 0.0) & (dt_sec <= threshold), dt_sec, 0.0
    )
    return points


def write_dwelling_cache(path, points, ranges_hash, start, end):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_ns=points["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        r_re=points["r_re"].to_numpy(float),
        mlat_deg=points["mlat_deg"].to_numpy(float),
        mlt_hour=points["mlt_hour"].to_numpy(float),
        l_90deg=points["l_90deg"].to_numpy(float),
        dwell_sec=points["dwell_sec"].to_numpy(float),
        ranges_sha256=np.asarray(ranges_hash),
        time_start=np.asarray(str(start)),
        time_end=np.asarray(str(end)),
        gap_threshold_sec=np.asarray(PLOT_CONFIG["orbit_gap_threshold_sec"]),
    )


def read_dwelling_cache(path, ranges_hash, start, end):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        compatible = (
            str(data["ranges_sha256"]) == ranges_hash
            and str(data["time_start"]) == str(start)
            and str(data["time_end"]) == str(end)
            and float(data["gap_threshold_sec"])
            == float(PLOT_CONFIG["orbit_gap_threshold_sec"])
        )
        if not compatible:
            return None
        return pd.DataFrame({
            "time": pd.to_datetime(data["time_ns"], unit="ns"),
            "r_re": data["r_re"],
            "mlat_deg": data["mlat_deg"],
            "mlt_hour": data["mlt_hour"],
            "l_90deg": data["l_90deg"],
            "dwell_sec": data["dwell_sec"],
        })


def _validated_edges(values, name):
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError(f"{name} edges must be a finite one-dimensional array")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"{name} edges must be strictly increasing")
    return edges


def select_phase_ranges(master: pd.DataFrame, phase_mode: str) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(master.columns)
    if missing:
        raise KeyError(f"Master CSV lacks columns: {sorted(missing)}")
    if phase_mode not in PHASES:
        raise ValueError(f"Unknown phase mode: {phase_mode}")
    phase = master.loc[master["phase_mode"].eq(phase_mode)].copy()
    if phase.empty:
        raise ValueError(f"Master CSV has no rows for phase_mode={phase_mode}")
    duplicate = phase["range_id"].duplicated(keep=False)
    if duplicate.any():
        ids = sorted(phase.loc[duplicate, "range_id"].astype(int).unique())
        raise ValueError(f"Duplicate range rows for phase_mode={phase_mode}: {ids}")
    allowed_status = {"ok", "insufficient_data"}
    if phase["status"].isna().any():
        raise ValueError("Completed range has a missing phase status")
    unexpected = sorted(set(phase["status"]) - allowed_status)
    if unexpected:
        raise ValueError(f"Unexpected phase status: {unexpected}")
    phase["range_id"] = phase["range_id"].astype(int)
    phase["range_duration_sec"] = pd.to_numeric(
        phase["range_duration_sec"], errors="coerce"
    )
    return phase


def _bin_indices(l_value, coordinate_value, duration, l_edges, coordinate_edges):
    finite = (
        np.isfinite(l_value) & np.isfinite(coordinate_value)
        & np.isfinite(duration) & (duration > 0)
    )
    in_domain = (
        finite & (l_value >= l_edges[0]) & (l_value < l_edges[-1])
        & (coordinate_value >= coordinate_edges[0])
        & (coordinate_value < coordinate_edges[-1])
    )
    l_bin = np.searchsorted(l_edges, l_value[in_domain], side="right") - 1
    coordinate_bin = (
        np.searchsorted(coordinate_edges, coordinate_value[in_domain], side="right") - 1
    )
    return finite, in_domain, l_bin.astype(int), coordinate_bin.astype(int)


def _in_wrapped_mlt_sector(values, sector=None):
    start, end = PLOT_CONFIG["l_mlat_mlt_sector_hour"] if sector is None else sector
    if not (0.0 <= start < 24.0 and 0.0 <= end < 24.0) or start == end:
        raise ValueError("MLT sector boundaries must differ and lie in [0, 24)")
    mlt = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = np.isfinite(mlt)
    mlt = np.mod(mlt, 24.0)
    if start < end:
        return finite & (mlt >= start) & (mlt < end)
    return finite & ((mlt >= start) | (mlt < end))


def aggregate_dwelling(points, coordinate_system, l_edges=None, coordinate_edges=None):
    spec = POSITION_COLUMNS[coordinate_system]
    l_edges = _validated_edges(
        PLOT_CONFIG["l_edges"] if l_edges is None else l_edges, "L"
    )
    coordinate_edges = _validated_edges(
        PLOT_CONFIG[spec["edges_key"]] if coordinate_edges is None else coordinate_edges,
        spec["coordinate_name"],
    )
    n_orbit_samples = len(points)
    n_mlt_sector_excluded = 0
    if coordinate_system == "l_mlat":
        sector_mask = _in_wrapped_mlt_sector(points["mlt_hour"])
        n_mlt_sector_excluded = int((~sector_mask).sum())
        points = points.loc[sector_mask]

    l_value = pd.to_numeric(points["l_90deg"], errors="coerce").to_numpy(float)
    coordinate_value = pd.to_numeric(
        points[spec["dwelling_column"]], errors="coerce"
    ).to_numpy(float)
    duration = pd.to_numeric(points["dwell_sec"], errors="coerce").to_numpy(float)
    finite, in_domain, l_bin, coordinate_bin = _bin_indices(
        l_value, coordinate_value, duration, l_edges, coordinate_edges
    )
    shape = (len(l_edges) - 1, len(coordinate_edges) - 1)
    dwelling = np.zeros(shape, dtype=float)
    samples = np.zeros(shape, dtype=int)
    np.add.at(dwelling, (l_bin, coordinate_bin), duration[in_domain])
    np.add.at(samples, (l_bin, coordinate_bin), 1)
    return dwelling, samples, {
        "n_orbit_samples": int(n_orbit_samples),
        "n_binned_orbit_samples": int(in_domain.sum()),
        "n_invalid_or_gap_samples": int((~finite).sum()),
        "n_outside_plot_domain": int((finite & ~in_domain).sum()),
        "n_orbit_samples_outside_mlt_sector": n_mlt_sector_excluded,
    }


def aggregate_occurrence(
    phase, points, coordinate_system, l_edges=None, coordinate_edges=None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Combine all-period spacecraft dwelling time with range-based KAW time."""
    if coordinate_system not in POSITION_COLUMNS:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")
    spec = POSITION_COLUMNS[coordinate_system]
    l_edges = _validated_edges(
        PLOT_CONFIG["l_edges"] if l_edges is None else l_edges, "L"
    )
    phase_mode = str(phase["phase_mode"].iloc[0])
    coordinate_edges = _validated_edges(
        PLOT_CONFIG[spec["edges_key"]] if coordinate_edges is None else coordinate_edges,
        spec["coordinate_name"],
    )
    dwelling, dwell_samples, diagnostics = aggregate_dwelling(
        points, coordinate_system, l_edges, coordinate_edges
    )
    n_phase_ranges = len(phase)
    n_mlt_sector_excluded = 0
    if coordinate_system == "l_mlat":
        sector_mask = _in_wrapped_mlt_sector(phase["range_MLT_circular_mean_hour"])
        n_mlt_sector_excluded = int((~sector_mask).sum())
        phase = phase.loc[sector_mask]

    l_value = pd.to_numeric(phase["range_L_90deg_mean"], errors="coerce").to_numpy(float)
    coordinate_value = pd.to_numeric(
        phase[spec["coordinate_column"]], errors="coerce"
    ).to_numpy(float)
    duration = pd.to_numeric(phase["range_duration_sec"], errors="coerce").to_numpy(float)
    finite, in_domain, l_bin, coordinate_bin = _bin_indices(
        l_value, coordinate_value, duration, l_edges, coordinate_edges
    )
    shape = dwelling.shape
    batch_duration = np.zeros(shape, dtype=float)
    kaw_duration = np.zeros(shape, dtype=float)
    n_batch = np.zeros(shape, dtype=int)
    n_kaw = np.zeros(shape, dtype=int)
    ok = phase["status"].eq("ok").to_numpy()[in_domain]
    selected_duration = duration[in_domain]
    np.add.at(batch_duration, (l_bin, coordinate_bin), selected_duration)
    np.add.at(n_batch, (l_bin, coordinate_bin), 1)
    np.add.at(kaw_duration, (l_bin[ok], coordinate_bin[ok]), selected_duration[ok])
    np.add.at(n_kaw, (l_bin[ok], coordinate_bin[ok]), 1)

    excess = kaw_duration > dwelling + 1e-6
    if np.any(excess):
        cells = np.argwhere(excess).tolist()
        raise ValueError(
            "KAW range duration exceeds spacecraft dwelling time in bins "
            f"{cells}; representative-position binning is inconsistent"
        )
    occurrence = np.divide(
        kaw_duration, dwelling, out=np.full(shape, np.nan), where=dwelling > 0
    )
    rows = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            rows.append({
                "coordinate_system": coordinate_system,
                "phase_mode": phase_mode,
                "l_bin": i, "coordinate_bin": j,
                "l_lower": l_edges[i], "l_upper": l_edges[i + 1],
                "coordinate_lower": coordinate_edges[j],
                "coordinate_upper": coordinate_edges[j + 1],
                "dwelling_duration_sec": dwelling[i, j],
                "kaw_duration_sec": kaw_duration[i, j],
                "batch_range_duration_sec": batch_duration[i, j],
                "dwelling_duration_hour": dwelling[i, j] / 3600.0,
                "kaw_duration_hour": kaw_duration[i, j] / 3600.0,
                "occurrence_probability": occurrence[i, j],
                "occurrence_percent": 100.0 * occurrence[i, j],
                "n_dwelling_samples": dwell_samples[i, j],
                "n_batch_ranges": n_batch[i, j],
                "n_kaw_ranges": n_kaw[i, j],
            })
    diagnostics.update({
        "n_phase_ranges": int(n_phase_ranges),
        "n_binned_ranges": int(in_domain.sum()),
        "n_invalid_range_position_or_duration": int((~finite).sum()),
        "n_ranges_outside_plot_domain": int((finite & ~in_domain).sum()),
        "n_ranges_outside_mlt_sector": n_mlt_sector_excluded,
    })
    return pd.DataFrame(rows), diagnostics


def table_to_arrays(table, l_edges, coordinate_edges):
    shape = (len(l_edges) - 1, len(coordinate_edges) - 1)
    return {
        column: table[column].to_numpy(float).reshape(shape)
        for column in (
            "dwelling_duration_hour", "kaw_duration_hour", "occurrence_percent"
        )
    }


def _masked_arrays(arrays):
    minimum_hour = float(PLOT_CONFIG["min_dwelling_duration_min"]) / 60.0
    covered = (
        (arrays["dwelling_duration_hour"] > 0.0)
        & (arrays["dwelling_duration_hour"] >= minimum_hour)
    )
    values = (
        arrays["dwelling_duration_hour"],
        arrays["kaw_duration_hour"],
        arrays["occurrence_percent"],
    )
    plotted = []
    for value in values:
        visible = covered.copy()
        if PLOT_CONFIG["color_scale"] == "log":
            visible &= value > 0.0
        plotted.append(np.ma.masked_where(~visible, value))
    return plotted, covered


def _colormaps():
    time_cmap = plt.get_cmap(PLOT_CONFIG["time_cmap"]).with_extremes(
        bad=PLOT_CONFIG["missing_color"]
    )
    probability_cmap = plt.get_cmap(
        PLOT_CONFIG["probability_cmap"]
    ).with_extremes(bad=PLOT_CONFIG["missing_color"])
    return time_cmap, probability_cmap


def _log_bounds(values, configured_limits):
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        automatic = (1.0, 10.0)
    else:
        automatic = (
            10.0 ** np.floor(np.log10(np.min(positive))),
            10.0 ** np.ceil(np.log10(np.max(positive))),
        )
    vmin = automatic[0] if configured_limits[0] is None else configured_limits[0]
    vmax = automatic[1] if configured_limits[1] is None else configured_limits[1]
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0:
        raise ValueError("Log color limits must be finite and strictly positive")
    if vmax <= vmin:
        vmax = 10.0 * vmin
    return float(vmin), float(vmax)


def _norms(arrays, covered):
    scale = PLOT_CONFIG["color_scale"]
    if scale == "linear":
        valid = arrays["dwelling_duration_hour"][covered]
        vmax = max(float(np.max(valid)) if valid.size else 1.0, np.finfo(float).eps)
        return Normalize(0.0, vmax), Normalize(
            *PLOT_CONFIG["probability_limits_percent"]
        )
    if scale != "log":
        raise ValueError(f"Unknown color scale: {scale}")
    time_values = np.concatenate([
        arrays["dwelling_duration_hour"][covered],
        arrays["kaw_duration_hour"][covered],
    ])
    probability_values = arrays["occurrence_percent"][covered]
    return (
        LogNorm(*_log_bounds(time_values, PLOT_CONFIG["time_log_limits_hour"])),
        LogNorm(*_log_bounds(
            probability_values, PLOT_CONFIG["probability_log_limits_percent"]
        )),
    )


def _panel_labels():
    return (
        "(a) Dwelling time of Arase",
        "(b) KAW observation time",
        "(c) Occurrence probability",
    )


def add_meridional_earth(ax):
    ax.add_patch(Wedge(
        (0, 0), 1.0, -90, 90,
        facecolor=PLOT_CONFIG["earth_nightside_color"], edgecolor="none", zorder=5,
    ))
    ax.add_patch(Wedge(
        (0, 0), 1.0, 90, 270,
        facecolor=PLOT_CONFIG["earth_dayside_color"], edgecolor="none", zorder=5,
    ))
    ax.add_patch(Circle(
        (0, 0), 1.0, facecolor="none",
        edgecolor=PLOT_CONFIG["earth_edgecolor"], linewidth=1.0, zorder=6,
    ))


def _dipole_mesh(l_edges, mlat_edges):
    l_grid, latitude_grid = np.meshgrid(l_edges, np.deg2rad(mlat_edges))
    radius = l_grid * np.cos(latitude_grid) ** 2
    return radius * np.cos(latitude_grid), radius * np.sin(latitude_grid)


def add_meridional_grid(ax, l_edges, mlat_edges):
    """Draw dipole-coordinate L and MLAT grid lines with sparse labels."""
    color = PLOT_CONFIG["grid_color"]
    latitude_dense = np.linspace(mlat_edges[0], mlat_edges[-1], 401)
    latitude_rad = np.deg2rad(latitude_dense)
    artists = []
    for l_value in l_edges:
        radius = l_value * np.cos(latitude_rad) ** 2
        artists.extend(ax.plot(
            radius * np.cos(latitude_rad), radius * np.sin(latitude_rad),
            color=color, alpha=PLOT_CONFIG["grid_major_alpha"],
            linewidth=PLOT_CONFIG["grid_major_linewidth"], zorder=3,
        ))

    l_dense = np.linspace(l_edges[0], l_edges[-1], 241)
    for latitude in mlat_edges:
        latitude_rad = np.deg2rad(latitude)
        radius = l_dense * np.cos(latitude_rad) ** 2
        major = np.isclose(np.mod(abs(latitude), 10.0), 0.0)
        artists.extend(ax.plot(
            radius * np.cos(latitude_rad), radius * np.sin(latitude_rad),
            color=color,
            alpha=PLOT_CONFIG["grid_major_alpha"] if major else PLOT_CONFIG["grid_minor_alpha"],
            linewidth=PLOT_CONFIG["grid_major_linewidth"] if major else PLOT_CONFIG["grid_minor_linewidth"],
            zorder=3,
        ))

    label_size = max(PLOT_CONFIG["font_size"] - 6, 7)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.4}
    for l_value in l_edges[::2]:
        artists.append(ax.text(
            l_value, -0.13, f"{l_value:g}", ha="center", va="top",
            fontsize=label_size, color="black", bbox=label_box, zorder=4,
        ))
    outer_l = l_edges[-1]
    for latitude in mlat_edges:
        if latitude == 0.0 or not np.isclose(np.mod(abs(latitude), 10.0), 0.0):
            continue
        if np.isclose(latitude, mlat_edges[0]) or np.isclose(latitude, mlat_edges[-1]):
            continue
        latitude_rad = np.deg2rad(latitude)
        radius = outer_l * np.cos(latitude_rad) ** 2
        x_value = 1.012 * radius * np.cos(latitude_rad)
        z_value = 1.012 * radius * np.sin(latitude_rad)
        artists.append(ax.text(
            x_value, z_value, f"{latitude:+g}°", ha="left", va="center",
            fontsize=label_size, color="black", bbox=label_box, zorder=4,
        ))
    return artists


def plot_l_mlat(table, phase_mode, date_label, output_base, show=False):
    l_edges = _validated_edges(PLOT_CONFIG["l_edges"], "L")
    mlat_edges = _validated_edges(PLOT_CONFIG["mlat_edges_deg"], "MLAT")
    arrays = table_to_arrays(table, l_edges, mlat_edges)
    plotted, covered = _masked_arrays(arrays)
    time_norm, probability_norm = _norms(arrays, covered)
    time_cmap, probability_cmap = _colormaps()
    x_edge, z_edge = _dipole_mesh(l_edges, mlat_edges)
    x_limit = float(np.nanmax(x_edge)) * 1.04
    z_limit = float(np.nanmax(np.abs(z_edge))) * 1.06
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    fig, axes = plt.subplots(
        1, 3, figsize=PLOT_CONFIG["figure_size_meridional"], dpi=PLOT_CONFIG["dpi"]
    )
    meshes = []
    for index, ax in enumerate(axes):
        probability = index == 2
        mesh = ax.pcolormesh(
            x_edge, z_edge, plotted[index].T,
            cmap=probability_cmap if probability else time_cmap,
            norm=probability_norm if probability else time_norm,
            shading="flat",
        )
        meshes.append(mesh)
        add_meridional_grid(ax, l_edges, mlat_edges)
        add_meridional_earth(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.15, x_limit)
        ax.set_ylim(-z_limit, z_limit)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_title(_panel_labels()[index])
        ax.set_xlabel(r"Nightside distance [$R_{\mathrm{E}}$]")
        label = "Occurrence probability [%]" if probability else "Time [hour]"
        fig.colorbar(mesh, ax=ax, pad=0.025, label=label)
    axes[0].set_ylabel(r"Northward distance [$R_{\mathrm{E}}$]")
    fig.suptitle(
        f"Arase KAW occurrence in L-MLAT: {PHASES[phase_mode]}\n{date_label}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_figure(fig, output_base, show)


def add_polar_earth(ax):
    ax.bar(
        0.0, 1.0, width=np.pi, bottom=0.0,
        color=PLOT_CONFIG["earth_nightside_color"], edgecolor="none", zorder=5,
    )
    ax.bar(
        np.pi, 1.0, width=np.pi, bottom=0.0,
        color=PLOT_CONFIG["earth_dayside_color"], edgecolor="none", zorder=5,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(
        theta, np.ones_like(theta), color=PLOT_CONFIG["earth_edgecolor"],
        linewidth=1.0, zorder=6,
    )


def plot_l_mlt(table, phase_mode, date_label, output_base, show=False):
    l_edges = _validated_edges(PLOT_CONFIG["l_edges"], "L")
    mlt_edges = _validated_edges(PLOT_CONFIG["mlt_edges_hour"], "MLT")
    arrays = table_to_arrays(table, l_edges, mlt_edges)
    plotted, covered = _masked_arrays(arrays)
    time_norm, probability_norm = _norms(arrays, covered)
    time_cmap, probability_cmap = _colormaps()
    theta_edges = 2.0 * np.pi * mlt_edges / 24.0
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    fig, axes = plt.subplots(
        1, 3, figsize=PLOT_CONFIG["figure_size_polar"], dpi=PLOT_CONFIG["dpi"],
        subplot_kw={"projection": "polar"},
    )
    for index, ax in enumerate(axes):
        probability = index == 2
        mesh = ax.pcolormesh(
            theta_edges, l_edges, plotted[index],
            cmap=probability_cmap if probability else time_cmap,
            norm=probability_norm if probability else time_norm,
            shading="flat",
        )
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        major_hours = np.array([0, 6, 12, 18])
        minor_hours = np.setdiff1d(np.arange(24), major_hours)
        ax.set_xticks(2.0 * np.pi * major_hours / 24.0)
        ax.set_xticklabels(["00", "06", "12", "18"])
        ax.set_xticks(2.0 * np.pi * minor_hours / 24.0, minor=True)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_ylim(0.0, l_edges[-1])
        ax.set_yticks(np.arange(2.0, l_edges[-1], 2.0))
        add_polar_earth(ax)
        ax.set_title(_panel_labels()[index], pad=23)
        ax.grid(
            which="major", color=PLOT_CONFIG["grid_color"],
            alpha=PLOT_CONFIG["grid_major_alpha"],
            linewidth=PLOT_CONFIG["grid_major_linewidth"],
        )
        ax.grid(
            which="minor", axis="x", color=PLOT_CONFIG["grid_color"],
            alpha=PLOT_CONFIG["grid_minor_alpha"],
            linewidth=PLOT_CONFIG["grid_minor_linewidth"],
        )
        label = "Occurrence probability [%]" if probability else "Time [hour]"
        fig.colorbar(mesh, ax=ax, pad=0.12, shrink=0.72, label=label)
    fig.suptitle(
        f"Arase KAW occurrence in L-MLT: {PHASES[phase_mode]}\n{date_label}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save_figure(fig, output_base, show)


def _save_figure(fig, output_base: Path, show=False):
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
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--refresh-dwelling-cache", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--orbit-download-retries", type=int, default=ORBIT_DOWNLOAD_RETRIES
    )
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
    start, end = manifest_time_window(payload, ranges)
    date_label = f"{start:%Y-%m-%d} to {(end - pd.Timedelta(1, unit='ns')):%Y-%m-%d}"
    cache_path = output_dir / (
        f"arase_orbit_dwelling_{start:%Y%m%d}_{end:%Y%m%d}.npz"
    )
    points = None if args.refresh_dwelling_cache else read_dwelling_cache(
        cache_path, ranges_hash, start, end
    )
    if points is None:
        files = ensure_orbit_files(
            start, end, args.spedas_dir,
            retries=args.orbit_download_retries,
            allow_download=not args.no_download,
        )
        print(f"ORB daily files available: {len(files)}")
        points = load_orbit_dwelling_points(start, end, args.spedas_dir)
        write_dwelling_cache(cache_path, points, ranges_hash, start, end)
        print(f"Wrote dwelling cache: {cache_path}")
    else:
        print(f"Reused dwelling cache: {cache_path}")
    coverage_hour = points["dwell_sec"].sum() / 3600.0
    interval_hour = (end - start).total_seconds() / 3600.0
    print(
        f"ORB samples: {len(points)}; weighted coverage: {coverage_hour:.3f}/"
        f"{interval_hour:.3f} h ({100.0 * coverage_hour / interval_hour:.2f}%)"
    )

    master = pd.read_csv(master_path)
    phases = list(dict.fromkeys(args.phase or PHASES))
    written = []
    for phase_mode in phases:
        phase = select_phase_ranges(master, phase_mode)
        for coordinate_system in ("l_mlat", "l_mlt"):
            table, diagnostics = aggregate_occurrence(
                phase, points, coordinate_system
            )
            stem = (
                f"arase_kaw_occurrence_{coordinate_system}_{phase_mode}_"
                f"{start:%Y%m%d}_{(end - pd.Timedelta(1, unit='ns')):%Y%m%d}"
            )
            table_path = output_dir / f"{stem}.csv"
            table_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(table_path, index=False)
            written.append(table_path)
            plotter = plot_l_mlat if coordinate_system == "l_mlat" else plot_l_mlt
            written.extend(plotter(
                table, phase_mode, date_label, output_dir / stem, show=args.show
            ))
            print(
                f"{phase_mode}/{coordinate_system}: "
                f"ranges={diagnostics['n_binned_ranges']}/"
                f"{diagnostics['n_phase_ranges']}; "
                f"orbit_samples={diagnostics['n_binned_orbit_samples']}/"
                f"{diagnostics['n_orbit_samples']}; "
                f"range_invalid={diagnostics['n_invalid_range_position_or_duration']}; "
                f"range_outside={diagnostics['n_ranges_outside_plot_domain']}"
            )
    print(f"dataset: {key}")
    print(f"master: {master_path}")
    for path in written:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
