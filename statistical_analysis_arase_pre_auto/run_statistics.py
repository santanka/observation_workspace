#!/usr/bin/env python3
"""Single entrypoint for prepared Arase statistics, optional enrichment, and figures."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import batch_arase as batch
import plot_environment_parameter_statistics as environment
from prepare_electron_beta_statistics_parallel import positive_workers
from prepare_geomag_derivative_statistics import config_key, lags
import statistics_selection as selection
from plot_x0_kappa_relationship import relationship_key
from statistics_period import resolve_period


PRODUCTS = ("spatial", "latitude", "dependence", "x0_kappa", "x0_kappa_fit", "spectra", "occurrence", "event_counts")
DERIVATIVES = ("d_dst_dt", "d_ae_dt", "d_sym_h_dt", "d_al_dt")


def build_parser():
    # Reuse the parameter registry, period/block options and all bin-width options.
    parser = environment.build_parser()
    parser.description = __doc__
    parser.epilog = "Default: all products and parameters. Existing figures are skipped unless --overwrite is supplied."
    parser.add_argument("--product", choices=PRODUCTS, action="append")
    parser.add_argument("--workers", type=positive_workers, default=2, help="Parallel beta_e range processes (default: 2)")
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--difference-method", choices=("centered", "backward"), default="centered")
    parser.add_argument("--dst-lag-hours", type=float, default=1.)
    parser.add_argument("--ae-lag-minutes", type=float, default=5.)
    parser.add_argument("--symh-lag-minutes", type=float, default=5.)
    parser.add_argument("--al-lag-minutes", type=float, default=5.)
    parser.add_argument("--allow-network", action="store_true", help="Permit beta_e input loaders to acquire missing CDFs")
    parser.add_argument("--download-missing-orbit", action="store_true")
    parser.add_argument("--timeout", type=int, default=14400, help="Beta_e timeout seconds per range")
    parser.add_argument("--l-bin", type=float, action="append")
    parser.add_argument("--x0-bin-width-dex", type=float, default=.5,
                        help="X0-kappa relationship bin width in log10(X0)")
    parser.add_argument("--reference-kperp-rhoi", type=float, default=1., help="Reference amplitude for x0_kappa product")
    parser.add_argument("--pivot-q", type=float, action="append", help="X0-kappa fit diagnostics; defaults 3 and 10")
    parser.add_argument("--bootstrap-months", type=int, default=300, help="X0-kappa OLS month-bootstrap draws")
    parser.add_argument("--kperp-rhoi-min", type=float, default=1.)
    parser.add_argument("--kperp-rhoi-max", type=float, default=100.)
    parser.add_argument("--kperp-rhoi-points", type=int, default=201)
    parser.add_argument("--latitude-min-bin-count", type=int, default=1,
                        help="Latitude profile summary minimum; retains legacy default 1")
    parser.add_argument("--latitude-coordinate", choices=("l_mlat", "l_abs_mlat"), action="append")
    parser.add_argument("--independent-y-limits", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help="Use prepared inputs/enrichment only")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare missing inputs/enrichment without figures")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading CDFs or writing artifacts")
    return parser


def build_steps(args):
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / selection.selection_key(args.min_occupied_blocks)
    products = list(dict.fromkeys(args.product or PRODUCTS))
    parameters = list(dict.fromkeys(args.parameter or environment.PARAMETERS))
    period_options = ["--start-month", f"{period.start:%Y%m}", "--end-month", f"{period.last_day:%Y%m}"]
    selection_options = ["--min-occupied-blocks", str(args.min_occupied_blocks)]
    common = [*period_options, "--statistics-dir", str(root), *selection_options]
    phase_options = [item for phase in args.phase or [] for item in ("--phase", phase)]
    policy = "--overwrite" if args.overwrite else "--skip-existing"
    steps = []
    directory = Path(__file__).resolve().parent
    def add(name, script, options):
        steps.append({"name": name, "command": [sys.executable, str(directory / script), *options]})
    inputs_ready = all((root / "inputs" / name).is_file() for name in
                       ("annual_ranges.json", "arase_phase_master.csv", "provenance.json"))
    if not inputs_ready:
        if args.plot_only:
            raise FileNotFoundError(f"Prepared inputs missing: {root / 'inputs'}")
        add("prepare_inputs", "plot_yearly_statistics.py",
            [*period_options, *selection_options, "--output-dir", str(root), "--prepare-only"])
    derivative_path = args.derivative_summary or root / "environment_inputs/geomag_derivatives" / config_key(
        args.difference_method, args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes) / "range_summary.csv"
    if "dependence" in products and not args.plot_only:
        if any(p in parameters for p in DERIVATIVES) and args.derivative_summary is None:
            add("prepare_derivatives", "prepare_geomag_derivative_statistics.py", [*common,
                "--spedas-dir", str(args.spedas_dir), "--difference-method", args.difference_method,
                "--dst-lag-hours", str(args.dst_lag_hours), "--ae-lag-minutes", str(args.ae_lag_minutes),
                "--symh-lag-minutes", str(args.symh_lag_minutes), "--al-lag-minutes", str(args.al_lag_minutes)])
        if "beta_e" in parameters and args.beta_e_summary is None:
            add("prepare_beta_e", "prepare_electron_beta_statistics_parallel.py", [*common,
                "--spedas-dir", str(args.spedas_dir), "--workers", str(args.workers),
                "--timeout", str(args.timeout), *(["--allow-network"] if args.allow_network else [])])
    if not args.prepare_only:
        if "spatial" in products:
            add("plot_spatial", "plot_kappa_x0_statistics.py", [
                "--ranges", str(root / "inputs/annual_ranges.json"),
                "--master", str(root / "inputs/arase_phase_master.csv"),
                "--output-dir", str(output / "kappa_x0_statistics"), *selection_options,
                "--min-bin-count", str(args.min_bin_count), *phase_options, policy])
        if "latitude" in products:
            options = [*common, "--output-dir", str(output / "kappa_x0_latitude_profiles"),
                       "--min-bin-count", str(args.latitude_min_bin_count), *phase_options, policy]
            for value in args.l_bin or []:
                options.extend(["--l-bin", str(value)])
            for value in args.latitude_coordinate or []:
                options.extend(["--coordinate", value])
            if args.independent_y_limits:
                options.append("--independent-y-limits")
            add("plot_latitude", "plot_kappa_x0_latitude_profiles.py", options)
        if "dependence" in products:
            options = [*common, "--output-dir", str(output), "--derivative-summary", str(derivative_path),
                       *phase_options, policy]
            for parameter in parameters:
                options.extend(["--parameter", parameter])
            for name in ("min_bin_count", "min_valid_fraction", "log_bin_width_dex", "dst_bin_width_nt",
                         "ae_bin_width_nt", "symh_bin_width_nt", "al_bin_width_nt", "vsys_x_bin_width_kms", "ddst_bin_width_nt_per_hour",
                         "dae_bin_width_nt_per_hour", "dsymh_bin_width_nt_per_hour", "dal_bin_width_nt_per_hour",
                         "beta_e_summary"):
                value = getattr(args, name)
                if value is not None:
                    options.extend(["--" + name.replace("_", "-"), str(value)])
            add("plot_dependence", "plot_environment_parameter_statistics.py", options)
        if "x0_kappa" in products:
            add("plot_x0_kappa", "plot_x0_kappa_relationship.py", [*common,
                "--output-dir", str(output / relationship_key(args.reference_kperp_rhoi)),
                "--min-bin-count", str(args.min_bin_count),
                "--reference-kperp-rhoi", str(args.reference_kperp_rhoi),
                "--x0-bin-width-dex", str(args.x0_bin_width_dex), *phase_options, policy])
        if "x0_kappa_fit" in products:
            options = [*common, "--output-dir", str(output / "x0_kappa_fits"),
                       "--bootstrap-months", str(args.bootstrap_months), *phase_options, policy]
            for q in args.pivot_q or []:
                options.extend(["--pivot-q", str(q)])
            add("fit_x0_kappa", "fit_x0_kappa_relationship.py", options)
        if "spectra" in products:
            add("plot_spectra", "plot_reconstructed_spectra.py", [*common,
                "--output-dir", str(output / "reconstructed_spectra"),
                "--kperp-rhoi-min", str(args.kperp_rhoi_min), "--kperp-rhoi-max", str(args.kperp_rhoi_max),
                "--kperp-rhoi-points", str(args.kperp_rhoi_points), *phase_options, policy])
        for product in ("occurrence", "event_counts"):
            if product in products:
                options = [*common, "--output-dir", str(output), "--spedas-dir", str(args.spedas_dir),
                           *phase_options, policy]
                if product == "event_counts":
                    options.append("--counts-only")
                elif args.download_missing_orbit:
                    options.append("--download-missing-orbit")
                add("plot_" + product, "plot_yearly_occurrence.py", options)
    return period, root, output, steps


def validate(args):
    import numpy as np
    relationship_key(args.reference_kperp_rhoi)
    if args.timeout <= 0 or args.min_bin_count < 1 or args.latitude_min_bin_count < 1:
        raise ValueError("timeout and min-bin-count must be positive")
    if args.bootstrap_months < 0 or any(not np.isfinite(q) or q <= 0 or q == 1 for q in args.pivot_q or []):
        raise ValueError("Require bootstrap-months >= 0 and positive finite pivots other than 1")
    if (not np.isfinite(args.kperp_rhoi_min) or not np.isfinite(args.kperp_rhoi_max)
            or not 0 < args.kperp_rhoi_min < args.kperp_rhoi_max or args.kperp_rhoi_points < 2):
        raise ValueError("Invalid reconstructed-spectrum grid")
    if args.min_valid_fraction is not None and not 0 <= args.min_valid_fraction <= 1:
        raise ValueError("Invalid optional valid-fraction threshold")
    for name, value in vars(args).items():
        if name.endswith(("lag_hours", "lag_minutes", "width_nt", "width_dex", "width_nt_per_hour", "width_kms")):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
    lags(args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes)
    if args.l_bin and any(v not in range(3, 11) for v in args.l_bin):
        raise ValueError("L bin lower edges must be 3, 4, ..., 10")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)
    period, root, output, steps = build_steps(args)
    if args.dry_run:
        import shlex
        for step in steps:
            print(step["name"] + ": " + shlex.join(step["command"]))
        return 0
    if not any(s["name"] == "prepare_inputs" for s in steps):
        selection.load_prepared_inputs(root, period)
    output.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
              "overwrite": args.overwrite, "workers": args.workers,
              "code_sha256": batch.file_sha256(Path(__file__)),
              "grid": "non-colormap: minorticks_on; major/minor grid alpha=0.3", "steps": []}
    journal = output / "statistics_run.json"
    env = dict(os.environ, MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/arase_matplotlib")
    for step in steps:
        item = {**step, "status": "running"}
        record["steps"].append(item)
        journal.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Starting {step['name']}", flush=True)
        try:
            subprocess.run(step["command"], cwd=batch.ROOT, env=env, check=True)
        except BaseException:
            item["status"] = "failed_or_interrupted"
            journal.write_text(json.dumps(record, indent=2) + "\n")
            raise
        item["status"] = "complete"
        journal.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Completed statistics: {journal}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
