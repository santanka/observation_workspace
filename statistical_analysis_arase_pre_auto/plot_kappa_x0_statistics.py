#!/usr/bin/env python3
"""Plot range-based kappa/X0 in L-MLAT, L-|MLAT| and L-MLT."""

from __future__ import annotations

import argparse
from statistics_plotting import add_output_arguments, should_write
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

import batch_arase as batch
import plot_occurrence_probability as occurrence
from statistics_phases import PHASES, classify_phases
from statistics_selection import add_selection_argument, select_events


PLOT_CONFIG = {
    "l_edges": np.arange(3.0, 12.0, 1.0),
    "mlat_edges_deg": np.arange(-50.0, 55.0, 5.0),
    "abs_mlat_edges_deg": np.arange(0.0, 55.0, 5.0),
    "mlt_edges_hour": np.arange(0.0, 25.0, 1.0),
    "l_mlat_mlt_sector_hour": (18.0, 6.0),
    "statistic": "median",
    "min_bin_count": 5,
    "color_percentiles": (5.0, 95.0),
    "kappa_color_limits": {"E": None, "B": None, "S": None},
    "log10_x0_color_limits": {"E": None, "B": None, "S": None},
    "kappa_cmap": "viridis",
    "log10_x0_cmap": "viridis",
    "missing_color": "0.82",
    "figure_size_meridional": (17.0, 6.0),
    "figure_size_polar": (17.0, 6.2),
    "combined_figure_size_meridional": (17.0, 11.5),
    "combined_figure_size_polar": (17.0, 12.0),
    "dpi": 300,
    "font_size": 15,
}

COMPONENTS = ("E", "B", "S")
FAMILIES = {
    "kappa": {
        "columns": {component: f"kappa_{component}" for component in COMPONENTS},
        "title": r"$\kappa$",
        "label": lambda component: rf"$\kappa_{{\mathrm{{{component}}}}}$",
        "cmap_key": "kappa_cmap",
    },
    "log10_x0": {
        "columns": {component: f"log10_X0_{component}" for component in COMPONENTS},
        "title": r"$\log_{10}(X_0)$",
        "label": lambda component: rf"$\log_{{10}}(X_{{0,\mathrm{{{component}}}}})$",
        "cmap_key": "log10_x0_cmap",
    },
}

POSITION_COLUMNS = {
    "l_mlat": {
        "coordinate_column": "range_MLAT_deg_mean",
        "edges_key": "mlat_edges_deg",
        "coordinate_name": "MLAT",
    },
    "l_mlt": {
        "coordinate_column": "range_MLT_circular_mean_hour",
        "edges_key": "mlt_edges_hour",
        "coordinate_name": "MLT",
    },
    "l_abs_mlat": {
        "coordinate_column": "range_MLAT_deg_mean",
        "edges_key": "abs_mlat_edges_deg",
        "coordinate_name": "|MLAT|",
    },
}

SUMMARY_FIELDS = ("mean", "std", "median", "q16", "q84", "min", "max")


def resolve_paths(args, ranges_hash):
    key = batch.dataset_key(args.ranges, ranges_hash)
    state_dir = args.state_dir or batch.DEFAULT_STATE_ROOT / key
    master = args.master or state_dir / "aggregate" / "arase_phase_master.csv"
    output_dir = (
        args.output_dir
        or batch.DEFAULT_OUTPUT_ROOT / key / "kappa_x0_statistics"
    )
    return key, master, output_dir


def _validated_edges(values, name):
    return occurrence._validated_edges(values, name)


def _select_phase(master, phase_mode):
    master = classify_phases(master)
    required = {
        "phase_mode", "status", "range_L_90deg_mean",
        "range_MLAT_deg_mean", "range_MLT_circular_mean_hour",
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
    if selected.empty:
        raise ValueError(f"No status=ok rows for phase_mode={phase_mode}")
    return selected


def _coordinate_selection(frame, coordinate_system):
    selected = frame.copy()
    n_before = len(selected)
    if coordinate_system in ("l_mlat", "l_abs_mlat"):
        sector = PLOT_CONFIG["l_mlat_mlt_sector_hour"]
        mask = occurrence._in_wrapped_mlt_sector(
            selected["range_MLT_circular_mean_hour"], sector=sector
        )
        selected = selected.loc[mask].copy()
    return selected, n_before - len(selected)


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


def _spatial_rows(frame, coordinate_system):
    """Attach plot-bin indices and return rows inside the plotted domain."""
    spec = POSITION_COLUMNS[coordinate_system]
    l_edges = _validated_edges(PLOT_CONFIG["l_edges"], "L")
    coordinate_edges = _validated_edges(
        PLOT_CONFIG[spec["edges_key"]], spec["coordinate_name"]
    )
    selected, n_sector_excluded = _coordinate_selection(frame, coordinate_system)
    l_value = pd.to_numeric(
        selected["range_L_90deg_mean"], errors="coerce"
    ).to_numpy(float)
    coordinate_value = pd.to_numeric(
        selected[spec["coordinate_column"]], errors="coerce"
    ).to_numpy(float)
    if coordinate_system == "l_abs_mlat":
        coordinate_value = np.abs(coordinate_value)
    finite_position = np.isfinite(l_value) & np.isfinite(coordinate_value)
    in_domain = (
        finite_position
        & (l_value >= l_edges[0]) & (l_value < l_edges[-1])
        & (coordinate_value >= coordinate_edges[0])
        & (coordinate_value < coordinate_edges[-1])
    )
    selected = selected.loc[in_domain].copy()
    selected["_l_bin"] = (
        np.searchsorted(l_edges, l_value[in_domain], side="right") - 1
    )
    selected["_coordinate_bin"] = (
        np.searchsorted(
            coordinate_edges, coordinate_value[in_domain], side="right"
        ) - 1
    )
    diagnostics = {
        "n_status_ok_ranges": int(len(frame)),
        "n_ranges_outside_mlt_sector": int(n_sector_excluded),
        "n_ranges_with_finite_position": int(finite_position.sum()),
        "n_ranges_in_plot_domain": int(in_domain.sum()),
    }
    return selected, diagnostics


def aggregate_statistics(frame, phase_mode, coordinate_system, family_name):
    if coordinate_system not in POSITION_COLUMNS:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")
    if family_name not in FAMILIES:
        raise ValueError(f"Unknown parameter family: {family_name}")
    spec = POSITION_COLUMNS[coordinate_system]
    family = FAMILIES[family_name]
    l_edges = _validated_edges(PLOT_CONFIG["l_edges"], "L")
    coordinate_edges = _validated_edges(
        PLOT_CONFIG[spec["edges_key"]], spec["coordinate_name"]
    )
    selected, diagnostics = _spatial_rows(frame, coordinate_system)

    rows = []
    for component, column in family["columns"].items():
        numeric = pd.to_numeric(selected[column], errors="coerce")
        for l_bin in range(len(l_edges) - 1):
            for coordinate_bin in range(len(coordinate_edges) - 1):
                cell = (
                    selected["_l_bin"].eq(l_bin)
                    & selected["_coordinate_bin"].eq(coordinate_bin)
                )
                stats = _summary(numeric.loc[cell].to_numpy(float))
                if coordinate_system == "l_abs_mlat":
                    latitude = pd.to_numeric(selected[spec["coordinate_column"]], errors="coerce")
                    valid = cell & np.isfinite(numeric)
                    stats.update(
                        n_north=int((valid & latitude.gt(0)).sum()),
                        n_south=int((valid & latitude.lt(0)).sum()),
                        n_equator=int((valid & latitude.eq(0)).sum()),
                    )
                rows.append({
                    "phase_mode": phase_mode,
                    "coordinate_system": coordinate_system,
                    "family": family_name,
                    "component": component,
                    "source_column": column,
                    "l_bin": l_bin,
                    "coordinate_bin": coordinate_bin,
                    "l_lower": l_edges[l_bin],
                    "l_upper": l_edges[l_bin + 1],
                    "coordinate_lower": coordinate_edges[coordinate_bin],
                    "coordinate_upper": coordinate_edges[coordinate_bin + 1],
                    **stats,
                })
    return pd.DataFrame(rows), diagnostics


def _limits(values, percentiles=None):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    if percentiles is None:
        low = float(np.min(finite))
        high = float(np.max(finite))
    else:
        low, high = np.percentile(finite, percentiles).astype(float)
    if np.isclose(low, high):
        pad = max(abs(low) * 0.05, 0.5)
        return low - pad, high + pad
    if percentiles is None:
        pad = 0.03 * (high - low)
        return low - pad, high + pad
    return low, high


def determine_color_limits(tables):
    statistic = PLOT_CONFIG["statistic"]
    minimum = int(PLOT_CONFIG["min_bin_count"])
    if minimum < 1:
        raise ValueError("min_bin_count must be at least 1")
    percentiles = tuple(PLOT_CONFIG["color_percentiles"])
    if (
        len(percentiles) != 2
        or not 0.0 <= percentiles[0] < percentiles[1] <= 100.0
    ):
        raise ValueError("color_percentiles must satisfy 0 <= low < high <= 100")
    result = {}
    kappa_config = PLOT_CONFIG["kappa_color_limits"]
    result["kappa"] = {}
    for component in COMPONENTS:
        configured = (
            kappa_config[component]
            if isinstance(kappa_config, dict)
            else kappa_config
        )
        if configured is None:
            values = pd.concat([
                table.loc[
                    table["family"].eq("kappa")
                    & table["component"].eq(component)
                    & table["n"].ge(minimum),
                    statistic,
                ]
                for table in tables
            ], ignore_index=True)
            result["kappa"][component] = _limits(values, percentiles)
        else:
            result["kappa"][component] = tuple(configured)

    result["log10_x0"] = {}
    for component in COMPONENTS:
        configured = PLOT_CONFIG["log10_x0_color_limits"][component]
        if configured is None:
            values = pd.concat([
                table.loc[
                    table["family"].eq("log10_x0")
                    & table["component"].eq(component)
                    & table["n"].ge(minimum),
                    statistic,
                ]
                for table in tables
            ], ignore_index=True)
            result["log10_x0"][component] = _limits(values, percentiles)
        else:
            result["log10_x0"][component] = tuple(configured)
    return result


def _component_array(table, component, value_column, l_edges, coordinate_edges):
    selected = table.loc[table["component"].eq(component)].sort_values(
        ["l_bin", "coordinate_bin"]
    )
    shape = (len(l_edges) - 1, len(coordinate_edges) - 1)
    return selected[value_column].to_numpy(float).reshape(shape)


def _cmap(family_name):
    return plt.get_cmap(
        PLOT_CONFIG[FAMILIES[family_name]["cmap_key"]]
    ).with_extremes(bad=PLOT_CONFIG["missing_color"])


def _colorbar_extend(family_name, component):
    configured = PLOT_CONFIG[
        "kappa_color_limits" if family_name == "kappa" else "log10_x0_color_limits"
    ]
    if isinstance(configured, dict):
        configured = configured[component]
    low, high = PLOT_CONFIG["color_percentiles"]
    return "both" if configured is None and (low > 0.0 or high < 100.0) else "neither"


def _masked_component(table, component, l_edges, coordinate_edges):
    statistic = PLOT_CONFIG["statistic"]
    values = _component_array(
        table, component, statistic, l_edges, coordinate_edges
    )
    counts = _component_array(
        table, component, "n", l_edges, coordinate_edges
    )
    valid = np.isfinite(values) & (counts >= int(PLOT_CONFIG["min_bin_count"]))
    return np.ma.masked_where(~valid, values)


def _panel_title(family_name, component):
    statistic = PLOT_CONFIG["statistic"]
    label = FAMILIES[family_name]["label"](component)
    return f"{statistic} {label}"


def displayed_range_summary(frame, table, coordinate_system, family_name, component):
    """Summarize range values belonging to bins visible in the color map."""
    selected, _ = _spatial_rows(frame, coordinate_system)
    counts = table.loc[table["component"].eq(component), [
        "l_bin", "coordinate_bin", "n",
    ]].copy()
    counts["_shown"] = counts["n"].ge(int(PLOT_CONFIG["min_bin_count"]))
    selected = selected.merge(
        counts[["l_bin", "coordinate_bin", "_shown"]],
        left_on=["_l_bin", "_coordinate_bin"],
        right_on=["l_bin", "coordinate_bin"],
        how="left",
        validate="many_to_one",
    )
    column = FAMILIES[family_name]["columns"][component]
    values = pd.to_numeric(
        selected.loc[selected["_shown"].fillna(False), column], errors="coerce"
    ).to_numpy(float)
    return _summary(values)


def _summary_title(family_name, component, summary, panel_label=None):
    first = _panel_title(family_name, component)
    if panel_label is not None:
        first = f"({panel_label}) {first}"
    if summary["n"] == 0:
        second = "overall: no displayed ranges"
    else:
        second = (
            f"median = {summary['median']:.3g} "
            f"[{summary['q16']:.3g}, {summary['q84']:.3g}]"
        )
    return f"{first}\n{second}"


def _configure_polar_axis(ax, l_edges):
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
    occurrence.add_polar_earth(ax)
    ax.grid(
        which="major", color=occurrence.PLOT_CONFIG["grid_color"],
        alpha=occurrence.PLOT_CONFIG["grid_major_alpha"],
        linewidth=occurrence.PLOT_CONFIG["grid_major_linewidth"],
    )
    ax.grid(
        which="minor", axis="x", color=occurrence.PLOT_CONFIG["grid_color"],
        alpha=occurrence.PLOT_CONFIG["grid_minor_alpha"],
        linewidth=occurrence.PLOT_CONFIG["grid_minor_linewidth"],
    )


def _coordinate_title(coordinate_system):
    return {
        "l_mlat": "L-MLAT (18-06 MLT)",
        "l_abs_mlat": "L-|MLAT| (18-06 MLT; hemispheres pooled)",
        "l_mlt": "L-MLT",
    }[coordinate_system]


def _meridional_ylabel(coordinate_system):
    if coordinate_system == "l_abs_mlat":
        return r"Magnetic-axis distance $|z|$ [$R_{\mathrm{E}}$]"
    return r"Northward distance [$R_{\mathrm{E}}$]"


def _add_meridional_grid(ax, l_edges, coordinate_edges, coordinate_system):
    occurrence.add_meridional_grid(ax, l_edges, coordinate_edges)
    if coordinate_system == "l_abs_mlat":
        for label in ax.texts:
            if label.get_text().endswith("°"):
                label.set_text(label.get_text().lstrip("+"))


def plot_statistics(
    table, phase_mode, coordinate_system, family_name, color_limits,
    date_label, output_base, show=False,
):
    l_edges = _validated_edges(PLOT_CONFIG["l_edges"], "L")
    coordinate_edges = _validated_edges(
        PLOT_CONFIG[POSITION_COLUMNS[coordinate_system]["edges_key"]],
        POSITION_COLUMNS[coordinate_system]["coordinate_name"],
    )
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    subplot_kw = {"projection": "polar"} if coordinate_system == "l_mlt" else {}
    figure_size = (
        PLOT_CONFIG["figure_size_polar"]
        if coordinate_system == "l_mlt"
        else PLOT_CONFIG["figure_size_meridional"]
    )
    fig, axes = plt.subplots(
        1, 3, figsize=figure_size, dpi=PLOT_CONFIG["dpi"],
        subplot_kw=subplot_kw,
    )
    cmap = _cmap(family_name)
    if coordinate_system in ("l_mlat", "l_abs_mlat"):
        x_edge, z_edge = occurrence._dipole_mesh(l_edges, coordinate_edges)
        x_limit = float(np.nanmax(x_edge)) * 1.04
        z_limit = float(np.nanmax(np.abs(z_edge))) * 1.06
    else:
        theta_edges = 2.0 * np.pi * coordinate_edges / 24.0

    for ax, component in zip(axes, COMPONENTS):
        plotted = _masked_component(
            table, component, l_edges, coordinate_edges
        )
        norm = Normalize(*color_limits[component])
        if coordinate_system in ("l_mlat", "l_abs_mlat"):
            mesh = ax.pcolormesh(
                x_edge, z_edge, plotted.T, cmap=cmap, norm=norm, shading="flat"
            )
            _add_meridional_grid(ax, l_edges, coordinate_edges, coordinate_system)
            occurrence.add_meridional_earth(ax)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.15, x_limit)
            ax.set_ylim(-1.15 if coordinate_system == "l_abs_mlat" else -z_limit, z_limit)
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
            ax.set_xlabel(r"Nightside distance [$R_{\mathrm{E}}$]")
        else:
            mesh = ax.pcolormesh(
                theta_edges, l_edges, plotted, cmap=cmap, norm=norm,
                shading="flat",
            )
            _configure_polar_axis(ax, l_edges)
        ax.set_title(_panel_title(family_name, component), pad=23)
        fig.colorbar(
            mesh, ax=ax,
            pad=0.12 if coordinate_system == "l_mlt" else 0.025,
            extend=_colorbar_extend(family_name, component),
        )
    if coordinate_system in ("l_mlat", "l_abs_mlat"):
        axes[0].set_ylabel(_meridional_ylabel(coordinate_system))
    family_title = FAMILIES[family_name]["title"]
    coordinate_title = _coordinate_title(coordinate_system)
    fig.suptitle(
        f"Arase {family_title} statistics in {coordinate_title}: "
        f"{PHASES[phase_mode]}\n{date_label}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return paths


def plot_combined_statistics(
    frame, tables, phase_mode, coordinate_system, color_limits,
    date_label, output_base, show=False,
):
    """Plot log10(X0) above kappa in one labeled 2-by-3 figure."""
    l_edges = _validated_edges(PLOT_CONFIG["l_edges"], "L")
    coordinate_edges = _validated_edges(
        PLOT_CONFIG[POSITION_COLUMNS[coordinate_system]["edges_key"]],
        POSITION_COLUMNS[coordinate_system]["coordinate_name"],
    )
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    subplot_kw = {"projection": "polar"} if coordinate_system == "l_mlt" else {}
    figure_size = (
        PLOT_CONFIG["combined_figure_size_polar"]
        if coordinate_system == "l_mlt"
        else PLOT_CONFIG["combined_figure_size_meridional"]
    )
    fig, axes = plt.subplots(
        2, 3, figsize=figure_size, dpi=PLOT_CONFIG["dpi"],
        subplot_kw=subplot_kw,
    )
    if coordinate_system in ("l_mlat", "l_abs_mlat"):
        x_edge, z_edge = occurrence._dipole_mesh(l_edges, coordinate_edges)
        x_limit = float(np.nanmax(x_edge)) * 1.04
        z_limit = float(np.nanmax(np.abs(z_edge))) * 1.06
    else:
        theta_edges = 2.0 * np.pi * coordinate_edges / 24.0

    panel_index = 0
    for row, family_name in enumerate(("log10_x0", "kappa")):
        table = tables[family_name]
        cmap = _cmap(family_name)
        for column, component in enumerate(COMPONENTS):
            ax = axes[row, column]
            plotted = _masked_component(
                table, component, l_edges, coordinate_edges
            )
            norm = Normalize(*color_limits[family_name][component])
            if coordinate_system in ("l_mlat", "l_abs_mlat"):
                mesh = ax.pcolormesh(
                    x_edge, z_edge, plotted.T, cmap=cmap, norm=norm,
                    shading="flat",
                )
                _add_meridional_grid(ax, l_edges, coordinate_edges, coordinate_system)
                occurrence.add_meridional_earth(ax)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(-1.15, x_limit)
                ax.set_ylim(-1.15 if coordinate_system == "l_abs_mlat" else -z_limit, z_limit)
                ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
                ax.set_xlabel(r"Nightside distance [$R_{\mathrm{E}}$]")
            else:
                mesh = ax.pcolormesh(
                    theta_edges, l_edges, plotted, cmap=cmap, norm=norm,
                    shading="flat",
                )
                _configure_polar_axis(ax, l_edges)
            summary = displayed_range_summary(
                frame, table, coordinate_system, family_name, component
            )
            panel_label = chr(ord("a") + panel_index)
            ax.set_title(
                _summary_title(family_name, component, summary, panel_label),
                pad=23,
            )
            fig.colorbar(
                mesh, ax=ax,
                pad=0.12 if coordinate_system == "l_mlt" else 0.025,
                extend=_colorbar_extend(family_name, component),
            )
            panel_index += 1
    if coordinate_system in ("l_mlat", "l_abs_mlat"):
        axes[0, 0].set_ylabel(_meridional_ylabel(coordinate_system))
        axes[1, 0].set_ylabel(_meridional_ylabel(coordinate_system))
    coordinate_title = (
        _coordinate_title(coordinate_system)
    )
    fig.suptitle(
        f"Arase $X_0$ and $\\kappa$ statistics in {coordinate_title}: "
        f"{PHASES[phase_mode]}\n{date_label}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return paths


def positive_bin_count(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--min-bin-count must be at least 1")
    return count


def add_bin_count_argument(parser):
    parser.add_argument(
        "--min-bin-count", type=positive_bin_count, default=PLOT_CONFIG["min_bin_count"],
        help="Minimum finite range count per spatial bin to color (default: 5); CSV retains all bins.",
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_output_arguments(parser)
    add_selection_argument(parser)
    add_bin_count_argument(parser)
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    PLOT_CONFIG["min_bin_count"] = args.min_bin_count
    payload, ranges = batch.load_ranges(args.ranges)
    ranges_hash = batch.file_sha256(args.ranges)
    key, master_path, output_dir = resolve_paths(args, ranges_hash)
    if not master_path.is_file():
        raise FileNotFoundError(f"Master CSV not found: {master_path}")
    start, end = occurrence.manifest_time_window(payload, ranges)
    last_day = end - pd.Timedelta(1, unit="ns")
    date_label = (f"{start:%Y-%m-%d} to {last_day:%Y-%m-%d}; "
                  f"occupied 10 s blocks >= {args.min_occupied_blocks}; "
                  f"bin n >= {args.min_bin_count}")
    master = pd.read_csv(master_path)
    master = select_events(master, args.min_occupied_blocks)
    phases = list(dict.fromkeys(args.phase or PHASES))

    aggregated = {}
    diagnostics = {}
    phase_frames = {}
    for phase_mode in phases:
        phase = _select_phase(master, phase_mode)
        phase_frames[phase_mode] = phase
        for coordinate_system in POSITION_COLUMNS:
            for family_name in FAMILIES:
                key_tuple = (phase_mode, coordinate_system, family_name)
                table, diagnostic = aggregate_statistics(
                    phase, phase_mode, coordinate_system, family_name
                )
                aggregated[key_tuple] = table
                diagnostics[key_tuple] = diagnostic
    color_limits = determine_color_limits(list(aggregated.values()))

    written = []
    for key_tuple, table in aggregated.items():
        phase_mode, coordinate_system, family_name = key_tuple
        stem = (
            f"arase_{family_name}_statistics_{coordinate_system}_{phase_mode}_"
            f"{start:%Y%m%d}_{last_day:%Y%m%d}"
        )
        csv_path = output_dir / f"{stem}.csv"
        if not should_write(output_dir / stem, overwrite=not args.skip_existing):
            continue
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        written.append(csv_path)
        written.extend(plot_statistics(
            table, phase_mode, coordinate_system, family_name,
            color_limits[family_name], date_label, output_dir / stem,
            show=args.show,
        ))
        diagnostic = diagnostics[key_tuple]
        print(
            f"{phase_mode}/{coordinate_system}/{family_name}: "
            f"in_domain={diagnostic['n_ranges_in_plot_domain']}/"
            f"{diagnostic['n_status_ok_ranges']}; "
            f"outside_mlt={diagnostic['n_ranges_outside_mlt_sector']}"
        )

    for phase_mode in phases:
        for coordinate_system in POSITION_COLUMNS:
            stem = (
                f"arase_kappa_x0_statistics_{coordinate_system}_{phase_mode}_"
                f"{start:%Y%m%d}_{last_day:%Y%m%d}"
            )
            tables = {
                family_name: aggregated[
                    (phase_mode, coordinate_system, family_name)
                ]
                for family_name in FAMILIES
            }
            if not should_write(output_dir / stem, overwrite=not args.skip_existing,
                                extensions=(".png", ".pdf")):
                continue
            written.extend(plot_combined_statistics(
                phase_frames[phase_mode], tables, phase_mode, coordinate_system,
                color_limits, date_label, output_dir / stem, show=args.show,
            ))

    print(f"dataset: {key}")
    print(f"master: {master_path}")
    print(f"min_bin_count: {PLOT_CONFIG['min_bin_count']}")
    for path in written:
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
