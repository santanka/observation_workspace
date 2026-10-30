#!/usr/bin/env python3
"""Plot saved range-based KAW fits versus environmental parameters."""

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
import plot_ae_parameter_statistics as ae
import statistics_phases as phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period
from statistics_plotting import detail_grid, add_output_arguments, should_write
import statistics_plotting
import statistics_correlations as correlations


PARAMETERS = {
    "ae": {"prefix": "range_geomag_AE_nT", "label": "AE [nT]", "scale": "linear",
           "width": 200., "unit": "nT", "minimum": 0.},
    "ne": {"prefix": "range_number_density_HFA_cc", "label": r"$n_e$ [cm$^{-3}$] (HFA)",
           "scale": "log", "width": 0.5, "unit": "cm^-3"},
    "dst": {"prefix": "range_geomag_Dst_nT", "label": "Dst [nT]",
            "scale": "linear", "width": 20., "unit": "nT"},
    "beta_i": {"prefix": "range_ion_plasma_beta_HFA", "label": r"$\beta_i$ (HFA-based)",
               "scale": "log", "width": 0.5, "unit": "dimensionless"},
    "ti_te": {"prefix": "range_i_e_temp_ratio", "label": r"$T_i/T_e$",
              "scale": "log", "width": 0.5, "unit": "dimensionless"},
    "beta_e": {"prefix": "range_electron_plasma_beta_HFA", "label": r"$\beta_e$ (HFA-based)",
               "scale": "log", "width": 0.5, "unit": "dimensionless"},
    "d_dst_dt": {"prefix": "range_geomag_dDst_dt_nT_per_hour", "label": r"$d\mathrm{Dst}/dt$ [nT/h]",
                 "scale": "linear", "width": 10., "unit": "nT/h"},
    "d_ae_dt": {"prefix": "range_geomag_dAE_dt_nT_per_hour", "label": r"$d\mathrm{AE}/dt$ [nT/h]",
                "scale": "linear", "width": 600., "unit": "nT/h"},
    "sym_h": {"prefix": "range_geomag_SYM_H_nT", "label": "SYM-H [nT]",
              "scale": "linear", "width": 20., "unit": "nT"},
    "al": {"prefix": "range_geomag_AL_nT", "label": "AL [nT]",
           "scale": "linear", "width": 200., "unit": "nT"},
    "d_sym_h_dt": {"prefix": "range_geomag_dSYM_H_dt_nT_per_hour", "label": r"$d\mathrm{SYM-H}/dt$ [nT/h]",
                   "scale": "linear", "width": 60., "unit": "nT/h"},
    "d_al_dt": {"prefix": "range_geomag_dAL_dt_nT_per_hour", "label": r"$d\mathrm{AL}/dt$ [nT/h]",
                "scale": "linear", "width": 600., "unit": "nT/h"},
    "vsys_x": {"prefix": "range_perp_sys_speed_ms", "label": r"$V_{\mathrm{sys},x}$ (FAC) [km/s]",
               "scale": "linear", "width": 10., "unit": "km/s", "source_unit": "m/s", "source_scale": 1e-3,
               "definition": "signed FAC x (toroidal) system velocity; observed partial ion moment, no spacecraft subtraction"},
    "abs_vsys_x": {"prefix": "range_perp_sys_speed_ms", "label": r"$|V_{\mathrm{sys},x}|$ (FAC) [km/s]",
                   "x_label": r"$|\mathrm{median}_{\mathrm{range}}(V_{\mathrm{sys},x})|$ (FAC) [km/s]",
                   "scale": "linear", "width": 10., "unit": "km/s", "source_unit": "m/s", "source_scale": 1e-3,
                   "source_transform": "abs", "minimum": 0.,
                   "definition": "absolute value of saved range median of signed FAC x velocity; not median of instantaneous absolute velocity"},
}
DEFAULT_PARAMETERS = ("ne", "dst", "beta_i", "ti_te")
PLOT_CONFIG = {"figure_size": (17, 10.5), "dpi": 300, "font_size": 14}


def parameter_values(frame, parameter):
    spec = PARAMETERS[parameter]
    values = pd.to_numeric(frame[spec["prefix"] + "_median"], errors="coerce") * spec.get("source_scale", 1.)
    return values.abs() if spec.get("source_transform") == "abs" else values


def load_electron_beta(master, path, master_hash):
    from prepare_electron_beta_statistics import METHOD, PREFIX
    return load_derived_summary(master, path, master_hash, METHOD, (PREFIX,))


def load_derived_summary(master, path, master_hash, method, prefixes):
    metadata = json.loads(path.with_suffix(".provenance.json").read_text())
    if metadata["master_sha256"] != master_hash or metadata["method"] != method:
        raise ValueError("Derived source master/method mismatch")
    if batch.file_sha256(path) != metadata["summary_sha256"]:
        raise ValueError("Derived summary checksum mismatch")
    sidecar = pd.read_csv(path)
    if sidecar.range_uid.duplicated().any() or sidecar.range_uid.isna().any():
        raise ValueError("Derived range_uid must be unique and nonmissing")
    if not sidecar.status.eq("complete").all() or not sidecar.method.eq(method).all():
        raise ValueError("Incomplete or incompatible derived rows")
    if not set(master.range_uid).issubset(set(sidecar.range_uid)):
        raise ValueError("Derived summary missing selected ranges; complete preparation first")
    columns = ["range_uid"] + [c for c in sidecar if any(c.startswith(p + "_") for p in prefixes)]
    collisions = (set(columns) - {"range_uid"}) & set(master.columns)
    if collisions:
        raise ValueError(f"Derived columns already in master: {sorted(collisions)}")
    return master.merge(sidecar[columns], on="range_uid", how="left", validate="many_to_one")


def select_parameter(master, parameter, phase, minimum_valid_fraction=None):
    spec = PARAMETERS[parameter]
    column, fraction = spec["prefix"] + "_median", spec["prefix"] + "_valid_fraction"
    required = {column, "phase_mode", "status"}
    if minimum_valid_fraction is not None:
        required.add(fraction)
    required.update(c for f in ae.FAMILIES.values() for c in f["columns"].values())
    missing = required - set(master.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    frame = master.loc[master.phase_mode.eq(phase) & master.status.eq("ok")].copy()
    x = parameter_values(frame, parameter)
    finite = np.isfinite(x)
    quality = pd.Series(True, index=frame.index)
    if minimum_valid_fraction is not None:
        vf = pd.to_numeric(frame[fraction], errors="coerce")
        quality = np.isfinite(vf) & vf.between(minimum_valid_fraction, 1.)
    positive = x.gt(0) if spec["scale"] == "log" else pd.Series(True, index=frame.index)
    keep = finite & quality & positive
    if "minimum" in spec:
        keep &= x.ge(spec["minimum"])
    diagnostics = {"input_ranges": len(frame), "accepted_ranges": int(keep.sum()),
                   "nonfinite_median": int((~finite).sum()),
                   "invalid_or_low_valid_fraction": int((~quality).sum()),
                   "nonpositive_median": int((finite & ~positive).sum())}
    return frame.loc[keep].copy(), diagnostics


def make_edges(frames, parameter, width):
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Bin width must be finite and positive")
    spec = PARAMETERS[parameter]
    values = pd.concat([parameter_values(f, parameter) for f in frames], ignore_index=True).to_numpy(float)
    if not len(values):
        raise ValueError(f"No valid ranges for {parameter}")
    transformed = np.log10(values) if spec["scale"] == "log" else values
    first = int(np.floor(transformed.min() / width))
    last = int(np.floor(transformed.max() / width)) + 1
    if last - first > 10000:
        raise ValueError("Requested bin width creates more than 10000 bins")
    edges = np.arange(first, last + 1, dtype=float) * width
    return 10. ** edges if spec["scale"] == "log" else edges


def aggregate(frame, parameter, phase, edges, minimum_count):
    spec = PARAMETERS[parameter]
    x = parameter_values(frame, parameter).to_numpy(float)
    bins = np.searchsorted(edges, x, side="right") - 1
    inside = np.isfinite(x) & (x >= edges[0]) & (x < edges[-1])
    rows = []
    for family, config in ae.FAMILIES.items():
        for component, column in config["columns"].items():
            y = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            pair = inside & np.isfinite(y)
            correlation = correlations.pair_correlations(x[pair], y[pair], log_x=spec["scale"] == "log")
            overall = ae._summary(y[pair])
            for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
                stats = ae._summary(y[pair & (bins == index)])
                center = np.sqrt(lower * upper) if spec["scale"] == "log" else (lower + upper) / 2
                rows.append({"parameter": parameter, "phase_mode": phase, "family": family,
                             "component": component, "source_column": column,
                             "x_source_column": spec["prefix"] + "_median", "x_unit": spec["unit"],
                             "x_source_unit": spec.get("source_unit", spec["unit"]),
                             "x_source_scale": spec.get("source_scale", 1.),
                             "x_source_transform": spec.get("source_transform", "identity"),
                             "bin": index, "lower": lower, "upper": upper, "center": center,
                             **stats, "shown": stats["n"] >= minimum_count,
                             "overall_n": overall["n"], **correlation,
                             "pearson_x_transform": "log10" if spec["scale"] == "log" else "identity",
                             "pearson_y_coordinate": column})
    result = pd.DataFrame(rows)
    if parameter == "ae":
        # Preserve historical AE CSV columns while using the common renderer.
        legacy = ae.aggregate_ae_statistics(frame, phase, edges).rename(columns={"ae_bin": "bin"})
        extra = [c for c in legacy if c not in result]
        result = result.merge(legacy[["family", "component", "bin", *extra]],
                              on=["family", "component", "bin"], validate="one_to_one")
        result["ae_bin"] = result["bin"]
    return result


def plot(frame, table, parameter, phase, edges, limits, date_label, base):
    spec = PARAMETERS[parameter]
    plt.rcParams.update({"font.size": PLOT_CONFIG["font_size"]})
    fig, axes = plt.subplots(2, 3, figsize=PLOT_CONFIG["figure_size"], sharex=True)
    x = parameter_values(frame, parameter)
    for r, family in enumerate(("log10_x0", "kappa")):
        for c, component in enumerate(ae.COMPONENTS):
            ax = axes[r, c]
            y = pd.to_numeric(frame[ae.FAMILIES[family]["columns"][component]], errors="coerce")
            finite = np.isfinite(y)
            ax.scatter(x[finite], y[finite], s=22, alpha=.28, color="0.35",
                       edgecolors="none", label="range")
            group = table.loc[table.family.eq(family) & table.component.eq(component)]
            shown = group.shown & np.isfinite(group["median"])
            visible = group.loc[shown]
            color = ae.PLOT_CONFIG["component_colors"][component]
            ax.plot(group.center, group["median"].where(shown), color=color, lw=1.8)
            ax.errorbar(visible.center, visible["median"],
                        yerr=np.array([visible["median"] - visible.q16,
                                       visible.q84 - visible["median"]]),
                        fmt="o", color=color, ms=5, capsize=3, label="bin median [q16, q84]")
            for _, cell in visible.iterrows():
                ax.annotate(f"n={int(cell.n)}", (cell.center, cell["median"]),
                            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=color)
            label = ae.FAMILIES[family]["label"](component)
            ax.set_title(f"({chr(97 + 3*r+c)}) {label}; N = {int(finite.sum())}\n"
                         + correlations.correlation_label(group.iloc[0].spearman_rho, group.iloc[0].pearson_r,
                                                          log_x=spec["scale"] == "log"))
            ax.set_ylabel(label)
            ax.set_xscale(spec["scale"])
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(*limits[family][component])
            if r == 1:
                ax.set_xlabel(spec.get("x_label", "Range median " + spec["label"]))
            detail_grid(ax)
    fig.suptitle(f"Arase KAW parameters versus {spec['label']}: {phases.PHASES[phase]} (all MLT)\n{date_label}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, .92))
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    statistics_plotting.save_figure(fig, paths[0], dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    statistics_plotting.save_figure(fig, paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    add_output_arguments(parser)
    parser.add_argument("--statistics-dir", type=Path, help="Prepared period directory")
    parser.add_argument("--output-dir", type=Path, help="Override content parent (contains ne_statistics etc.)")
    parser.add_argument("--parameter", choices=tuple(PARAMETERS), action="append")
    parser.add_argument("--phase", choices=tuple(phases.PHASES), action="append")
    parser.add_argument("--min-bin-count", type=int, default=5)
    parser.add_argument("--min-valid-fraction", type=float, default=None,
                        help="Optional coverage screening; disabled by default")
    parser.add_argument("--log-bin-width-dex", type=float, default=.5)
    parser.add_argument("--dst-bin-width-nt", type=float, default=20.)
    parser.add_argument("--ae-bin-width-nt", type=float, default=200.)
    parser.add_argument("--vsys-x-bin-width-kms", type=float, default=10.)
    parser.add_argument("--beta-e-summary", type=Path, help="Prepared electron-beta range_summary.csv")
    parser.add_argument("--derivative-summary", type=Path, help="Prepared native-index derivative range_summary.csv")
    parser.add_argument("--ddst-bin-width-nt-per-hour", type=float, default=10.)
    parser.add_argument("--dae-bin-width-nt-per-hour", type=float, default=600.)
    parser.add_argument("--symh-bin-width-nt", type=float, default=20.)
    parser.add_argument("--al-bin-width-nt", type=float, default=200.)
    parser.add_argument("--dsymh-bin-width-nt-per-hour", type=float, default=60.)
    parser.add_argument("--dal-bin-width-nt-per-hour", type=float, default=600.)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.min_bin_count < 1 or (args.min_valid_fraction is not None and not 0 <= args.min_valid_fraction <= 1):
        raise ValueError("Require min-bin-count >= 1 and 0 <= min-valid-fraction <= 1")
    if any(not np.isfinite(w) or w <= 0 for w in (args.log_bin_width_dex, args.dst_bin_width_nt,
                                                  args.ddst_bin_width_nt_per_hour, args.dae_bin_width_nt_per_hour,
                                                  args.symh_bin_width_nt, args.al_bin_width_nt,
                                                  args.dsymh_bin_width_nt_per_hour, args.dal_bin_width_nt_per_hour,
                                                  args.ae_bin_width_nt, args.vsys_x_bin_width_kms)):
        raise ValueError("Bin widths must be finite and positive")
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks)
    master, provenance = selection.load_prepared_inputs(root, period)
    master = phases.classify_phases(master)
    selected = selection.select_events(master, args.min_occupied_blocks)
    requested_phases = list(dict.fromkeys(args.phase or phases.PHASES))
    for parameter in dict.fromkeys(args.parameter or DEFAULT_PARAMETERS):
        source_frame = selected
        beta_path = None
        derivative_path = None
        derivative_metadata = None
        if parameter == "beta_e":
            beta_path = args.beta_e_summary or root / "environment_inputs/beta_e/range_summary.csv"
            # Only requested phases must be covered; do not silently drop missing ranges.
            source_frame = load_electron_beta(selected.loc[selected.phase_mode.isin(requested_phases)],
                                             beta_path, provenance["annual_master_sha256"])
        elif parameter in ("d_dst_dt", "d_ae_dt", "d_sym_h_dt", "d_al_dt"):
            from prepare_geomag_derivative_statistics import METHOD, config_key
            derivative_path = args.derivative_summary or root / "environment_inputs/geomag_derivatives" / config_key() / "range_summary.csv"
            source_frame = load_derived_summary(selected.loc[selected.phase_mode.isin(requested_phases)],
                derivative_path, provenance["annual_master_sha256"], METHOD, (PARAMETERS[parameter]["prefix"],))
            derivative_metadata = json.loads(derivative_path.with_suffix(".provenance.json").read_text())
        frames, diagnostics = {}, {}
        for phase in requested_phases:
            frames[phase], diagnostics[phase] = select_parameter(source_frame, parameter, phase, args.min_valid_fraction)
        widths = {"vsys_x": args.vsys_x_bin_width_kms, "abs_vsys_x": args.vsys_x_bin_width_kms,
                  "ae": args.ae_bin_width_nt, "dst": args.dst_bin_width_nt, "d_dst_dt": args.ddst_bin_width_nt_per_hour,
                  "d_ae_dt": args.dae_bin_width_nt_per_hour, "sym_h": args.symh_bin_width_nt,
                  "al": args.al_bin_width_nt, "d_sym_h_dt": args.dsymh_bin_width_nt_per_hour,
                  "d_al_dt": args.dal_bin_width_nt_per_hour}
        width = widths.get(parameter, args.log_bin_width_dex)
        if not any(len(frame) for frame in frames.values()):
            raise ValueError(f"No valid ranges for {parameter}: {diagnostics}")
        edges = make_edges(list(frames.values()), parameter, width)
        limits = ae.determine_y_limits(list(frames.values()))
        directory = output / f"{parameter}_statistics"
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        date_label = (f"{period.start:%Y-%m-%d} to {period.last_day:%Y-%m-%d}; "
                      f"occupied 10 s blocks >= {args.min_occupied_blocks}; "
                      f"bin n >= {args.min_bin_count}")
        if derivative_metadata is not None:
            config = derivative_metadata["configuration"]
            lag_keys = {"d_ae_dt": "ae_lag_minutes", "d_sym_h_dt": "symh_lag_minutes", "d_al_dt": "al_lag_minutes"}
            lag = (f"{config['dst_lag_hours']:g} h" if parameter == "d_dst_dt"
                   else f"{config[lag_keys[parameter]]:g} min")
            stencil = f"+/- {lag}" if config["difference_method"] == "centered" else f"previous {lag}"
            date_label += f"\n{config['difference_method']} difference ({stencil}); native cadence, no smoothing"
        for phase, frame in frames.items():
            if frame.empty:
                print(f"Skip empty phase: {parameter}/{phase}", flush=True)
                continue
            table = aggregate(frame, parameter, phase, edges, args.min_bin_count)
            base = directory / f"arase_{parameter}_parameter_statistics_{phase}_{period.date_suffix}"
            if not should_write(base, overwrite=not args.skip_existing):
                continue
            table.to_csv(base.with_suffix(".csv"), index=False)
            written.append(base.with_suffix(".csv"))
            written.extend(plot(frame, table, parameter, phase, edges, limits, date_label, base))
            print(f"{parameter}/{phase}: {diagnostics[phase]}", flush=True)
        if not written:
            continue
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
            "parameter": parameter, "parameter_config": PARAMETERS[parameter],
            "min_bin_count": args.min_bin_count, "min_valid_fraction": args.min_valid_fraction,
            "valid_fraction_screening": args.min_valid_fraction is not None,
            "bin_width": width, "bin_edges": edges.tolist(), "mlt_sector": "all",
            "selection": selection.selection_metadata(master, args.min_occupied_blocks),
            "diagnostics": diagnostics, "plot_config": PLOT_CONFIG, "y_limits": limits,
            "grid_config": statistics_plotting.GRID_CONFIG,
            "master": str(root / "inputs/arase_phase_master.csv"),
            "master_sha256": provenance["annual_master_sha256"],
            "source_provenance_sha256": batch.file_sha256(root / "inputs/provenance.json"),
            "summary": "range-weighted median and q16-q84 distribution spread; not confidence interval",
            "correlation": "Spearman rho and Pearson r from individual accepted finite pairs; Pearson uses displayed coordinates (log10 x for log axis, saved log10_X0 or kappa for y); no covariance or independence correction",
            "code_sha256": {str(Path(m.__file__).resolve()): batch.file_sha256(Path(m.__file__))
                            for m in (ae, phases, selection, statistics_plotting, correlations)},
            "written": [str(p) for p in written],
        }
        metadata["code_sha256"][str(Path(__file__).resolve())] = batch.file_sha256(Path(__file__))
        if beta_path is not None:
            metadata["beta_e_summary"] = str(beta_path)
            metadata["beta_e_summary_sha256"] = batch.file_sha256(beta_path)
            metadata["beta_e_provenance_sha256"] = batch.file_sha256(beta_path.with_suffix(".provenance.json"))
        if derivative_path is not None:
            metadata["derivative_summary"] = str(derivative_path)
            metadata["derivative_summary_sha256"] = batch.file_sha256(derivative_path)
            metadata["derivative_provenance_sha256"] = batch.file_sha256(derivative_path.with_suffix(".provenance.json"))
            metadata["derivative_configuration"] = derivative_metadata["configuration"]
        (directory / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Wrote {len(written)} files: {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
