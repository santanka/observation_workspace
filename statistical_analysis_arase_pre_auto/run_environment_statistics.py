#!/usr/bin/env python3
"""Prepare beta_e and native geomagnetic derivatives, then render their dependencies."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import batch_arase as batch
from prepare_geomag_derivative_statistics import config_key, lags
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period


DEFAULT_PARAMETERS = ("ti_te", "beta_e", "d_dst_dt", "d_ae_dt", "sym_h", "al", "d_sym_h_dt", "d_al_dt")
PARAMETERS = ("ne", "dst", "beta_i", *DEFAULT_PARAMETERS)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--plot-output-dir", type=Path, help="Alternate content parent for sensitivity figures")
    parser.add_argument("--parameter", choices=PARAMETERS, action="append")
    parser.add_argument("--phase", choices=("all", "toward_ionosphere", "toward_magnetic_equator", "standing"), action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    mode.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-ranges", type=int)
    parser.add_argument("--difference-method", choices=("centered", "backward"), default="centered")
    parser.add_argument("--dst-lag-hours", type=float, default=1.)
    parser.add_argument("--ae-lag-minutes", type=float, default=5.)
    parser.add_argument("--symh-lag-minutes", type=float, default=5.)
    parser.add_argument("--al-lag-minutes", type=float, default=5.)
    parser.add_argument("--allow-network", action="store_true", help="Permit beta_e loaders to acquire missing inputs")
    parser.add_argument("--timeout", type=int, default=14400, help="Beta_e timeout seconds per range")
    parser.add_argument("--min-bin-count", type=int, default=5)
    parser.add_argument("--min-valid-fraction", type=float, default=None,
                        help="Optional coverage screening; disabled by default")
    parser.add_argument("--log-bin-width-dex", type=float, default=.5)
    parser.add_argument("--dst-bin-width-nt", type=float, default=20.)
    parser.add_argument("--ddst-bin-width-nt-per-hour", type=float, default=10.)
    parser.add_argument("--dae-bin-width-nt-per-hour", type=float, default=600.)
    parser.add_argument("--symh-bin-width-nt", type=float, default=20.)
    parser.add_argument("--al-bin-width-nt", type=float, default=200.)
    parser.add_argument("--dsymh-bin-width-nt-per-hour", type=float, default=60.)
    parser.add_argument("--dal-bin-width-nt-per-hour", type=float, default=600.)
    return parser


def commands(args, root, period):
    parameters = list(dict.fromkeys(args.parameter or DEFAULT_PARAMETERS))
    common = ["--start-month", f"{period.start:%Y%m}", "--end-month", f"{period.last_day:%Y%m}",
              "--statistics-dir", str(root), "--min-occupied-blocks", str(args.min_occupied_blocks)]
    steps = []
    script_root = Path(__file__).resolve().parent
    def step(name, script, options):
        steps.append((name, [sys.executable, str(script_root / script), *common, *options]))
    limit = ["--max-ranges", str(args.max_ranges)] if args.max_ranges else []
    plan = ["--plan-only"] if args.plan_only else []
    key = config_key(args.difference_method, args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes)
    derivative_summary = root / "environment_inputs/geomag_derivatives" / key / "range_summary.csv"
    if not args.plot_only:
        if any(p in parameters for p in ("d_dst_dt", "d_ae_dt", "d_sym_h_dt", "d_al_dt")):
            step("prepare_geomag_derivatives", "prepare_geomag_derivative_statistics.py", [
                "--spedas-dir", str(args.spedas_dir), "--difference-method", args.difference_method,
                "--dst-lag-hours", str(args.dst_lag_hours), "--ae-lag-minutes", str(args.ae_lag_minutes),
                "--symh-lag-minutes", str(args.symh_lag_minutes), "--al-lag-minutes", str(args.al_lag_minutes), *limit, *plan])
        if "beta_e" in parameters:
            step("prepare_beta_e", "prepare_electron_beta_statistics.py", [
                "--spedas-dir", str(args.spedas_dir), "--timeout", str(args.timeout), *limit, *plan,
                *(["--allow-network"] if args.allow_network else [])])
    if not args.prepare_only and not args.plan_only:
        options = ["--derivative-summary", str(derivative_summary)]
        if args.plot_output_dir:
            options.extend(["--output-dir", str(args.plot_output_dir)])
        for p in parameters:
            options.extend(["--parameter", p])
        for phase in args.phase or []:
            options.extend(["--phase", phase])
        for name in ("min_bin_count", "min_valid_fraction", "log_bin_width_dex", "dst_bin_width_nt",
                     "ddst_bin_width_nt_per_hour", "dae_bin_width_nt_per_hour", "symh_bin_width_nt",
                     "al_bin_width_nt", "dsymh_bin_width_nt_per_hour", "dal_bin_width_nt_per_hour"):
            value = getattr(args, name)
            if value is not None:
                options.extend(["--" + name.replace("_", "-"), str(value)])
        step("plot_dependencies", "plot_environment_parameter_statistics.py", options)
    return steps


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_ranges is not None and (args.max_ranges <= 0 or not (args.prepare_only or args.plan_only)):
        raise ValueError("--max-ranges requires --prepare-only or --plan-only and a positive count")
    # Validate widths and lags before any expensive preparation.
    import numpy as np
    values = (args.dst_lag_hours, args.ae_lag_minutes, args.log_bin_width_dex,
              args.symh_lag_minutes, args.al_lag_minutes, args.dst_bin_width_nt,
              args.ddst_bin_width_nt_per_hour, args.dae_bin_width_nt_per_hour,
              args.symh_bin_width_nt, args.al_bin_width_nt, args.dsymh_bin_width_nt_per_hour, args.dal_bin_width_nt_per_hour)
    if any(not np.isfinite(x) or x <= 0 for x in values) or args.timeout <= 0:
        raise ValueError("All lags, widths and timeout must be finite and positive")
    if args.min_bin_count < 1 or (args.min_valid_fraction is not None and not 0 <= args.min_valid_fraction <= 1):
        raise ValueError("Invalid bin count or valid fraction")
    lags(args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes)
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    selection.load_prepared_inputs(root, period)
    steps = commands(args, root, period)
    audit = root / "environment_inputs/pipeline"
    audit.mkdir(parents=True, exist_ok=True)
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "period": period.key,
                "code_sha256": batch.file_sha256(Path(__file__)), "steps": []}
    journal = audit / "latest_run.json"
    env = dict(os.environ, MPLBACKEND="Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp/arase_matplotlib")
    for name, command in steps:
        record = {"name": name, "command": command, "status": "running"}
        metadata["steps"].append(record)
        journal.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Starting {name}", flush=True)
        try:
            subprocess.run(command, cwd=batch.ROOT, env=env, check=True)
        except BaseException:
            record["status"] = "failed_or_interrupted"
            journal.write_text(json.dumps(metadata, indent=2) + "\n")
            raise
        record["status"] = "complete"
        journal.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {len(steps)} steps. Execution record: {journal}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
