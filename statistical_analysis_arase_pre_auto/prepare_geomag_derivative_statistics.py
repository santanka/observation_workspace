#!/usr/bin/env python3
"""Derive native-cadence Dst/AE rates from local OMNI CDFs, then summarize ranges."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import batch_arase as batch
from prepare_electron_beta_statistics import describe
import statistics_phases as phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period


METHOD = "native_omni_geomag_derivatives_v1"
SPECS = {
    "d_dst_dt": {"prefix": "range_geomag_dDst_dt_nT_per_hour", "product": "hourly",
                 "variable": "DST", "cadence": "1h", "index": "Dst"},
    "d_ae_dt": {"prefix": "range_geomag_dAE_dt_nT_per_hour", "product": "hro_1min",
                "variable": "AE_INDEX", "cadence": "1min", "index": "AE"},
    "d_sym_h_dt": {"prefix": "range_geomag_dSYM_H_dt_nT_per_hour", "product": "hro_1min",
                   "variable": "SYM_H", "cadence": "1min", "index": "SYM_H"},
    "d_al_dt": {"prefix": "range_geomag_dAL_dt_nT_per_hour", "product": "hro_1min",
                "variable": "AL_INDEX", "cadence": "1min", "index": "AL"},
}


def config_key(method="centered", dst_lag_hours=1., ae_lag_minutes=5., symh_lag_minutes=5., al_lag_minutes=5.):
    return (f"{method}_dst_{dst_lag_hours:g}h_ae_{ae_lag_minutes:g}min"
            f"_symh_{symh_lag_minutes:g}min_al_{al_lag_minutes:g}min").replace(".", "p")


def lags(dst_hours, ae_minutes, symh_minutes=5., al_minutes=5.):
    result = {"d_dst_dt": pd.Timedelta(hours=dst_hours), "d_ae_dt": pd.Timedelta(minutes=ae_minutes),
              "d_sym_h_dt": pd.Timedelta(minutes=symh_minutes), "d_al_dt": pd.Timedelta(minutes=al_minutes)}
    for key, lag in result.items():
        cadence = pd.Timedelta(SPECS[key]["cadence"])
        if lag <= pd.Timedelta(0) or lag.value % cadence.value:
            raise ValueError(f"{key} lag must be a positive integer multiple of {cadence}")
    return result


def file_inventory(spedas_dir, rows, lag_map, method):
    """Select latest cached version per month/half year, including derivative margins."""
    inventory = {}
    missing = {}
    for parameter, spec in SPECS.items():
        labels = set()
        for _, row in rows.iterrows():
            start = pd.Timestamp(row.range_start) - lag_map[parameter] - pd.Timedelta(spec["cadence"])
            end = pd.Timestamp(row.range_end) + (lag_map[parameter] if method == "centered" else pd.Timedelta(0)) + pd.Timedelta(spec["cadence"])
            for month in pd.period_range(start, end, freq="M"):
                labels.add(f"{month.year:04d}{month.month:02d}01" if spec["product"] == "hro_1min"
                           else f"{month.year:04d}{1 if month.month <= 6 else 7:02d}01")
        found, absent = [], []
        for label in sorted(labels):
            pattern = (f"omni_hro_1min_{label}_v*.cdf" if spec["product"] == "hro_1min"
                       else f"omni2_h0_mrg1hr_{label}_v*.cdf")
            candidates = list((Path(spedas_dir) / "omni" / spec["product"] / label[:4]).glob(pattern))
            if candidates:
                found.append(max(candidates, key=lambda p: int(re.search(r"_v(\d+)\.cdf$", p.name).group(1))))
            else:
                absent.append(pattern)
        inventory[parameter] = found
        missing[parameter] = absent
    return inventory, missing


def read_native(paths, variable):
    from cdflib import CDF, cdfepoch
    pieces = []
    for path in paths:
        with CDF(path) as cdf:
            info = cdf.cdf_info()
            if variable not in info.rVariables + info.zVariables:
                raise KeyError(f"{path} lacks {variable}")
            attributes = cdf.varattsget(variable)
            epoch = attributes.get("DEPEND_0", "Epoch")
            times = pd.DatetimeIndex(cdfepoch.to_datetime(cdf.varget(epoch)))
            values = np.asarray(cdf.varget(variable), dtype=float).reshape(-1)
            valid = np.isfinite(values)
            if "FILLVAL" in attributes:
                valid &= values != float(np.asarray(attributes["FILLVAL"]).squeeze())
            for field, compare in (("VALIDMIN", np.greater_equal), ("VALIDMAX", np.less_equal)):
                if field in attributes:
                    valid &= compare(values, float(np.asarray(attributes[field]).squeeze()))
            values[~valid] = np.nan
            piece = pd.Series(values, index=times)
            pieces.append(piece.loc[~piece.index.isna()])
    if not pieces:
        raise FileNotFoundError(f"No local CDFs for {variable}")
    series = pd.concat(pieces).sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"Duplicate native timestamps for {variable}; inspect cached sources")
    return series


def differentiate(series, cadence, lag, method="centered"):
    """No smoothing or gap bridging; every native sample in the stencil must be valid."""
    cadence, lag = pd.Timedelta(cadence), pd.Timedelta(lag)
    if series.empty or lag <= pd.Timedelta(0) or lag.value % cadence.value:
        raise ValueError("Invalid native series or derivative lag")
    if method not in ("centered", "backward"):
        raise ValueError("Unknown difference method")
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError("Native timestamps must be unique and sorted")
    time_ns = series.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if np.any(time_ns % cadence.value):
        raise ValueError("Native timestamps do not lie on the expected UTC cadence grid")
    grid = pd.date_range(series.index[0], series.index[-1], freq=cadence)
    source = series.reindex(grid).astype(float).replace([np.inf, -np.inf], np.nan)
    steps = int(lag / cadence)
    hours = lag / pd.Timedelta(hours=1)
    if method == "centered":
        rate = (source.shift(-steps) - source.shift(steps)) / (2 * hours)
        complete = source.notna().rolling(2*steps+1, center=True, min_periods=2*steps+1).sum().eq(2*steps+1)
    else:
        rate = (source - source.shift(steps)) / hours
        complete = source.notna().rolling(steps+1, min_periods=steps+1).sum().eq(steps+1)
    return rate.where(complete)


def sample_rates(rate, target, cadence):
    """Match nearest native rate with half-cadence tolerance, retaining invalid native slots."""
    target = pd.DatetimeIndex(target)
    if target.empty or target.has_duplicates:
        raise ValueError("Saved range time grid is empty or duplicated")
    return rate.reindex(target, method="nearest", tolerance=pd.Timedelta(cadence)/2).to_numpy(float)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--difference-method", choices=("centered", "backward"), default="centered")
    parser.add_argument("--dst-lag-hours", type=float, default=1.)
    parser.add_argument("--ae-lag-minutes", type=float, default=5.)
    parser.add_argument("--symh-lag-minutes", type=float, default=5.)
    parser.add_argument("--al-lag-minutes", type=float, default=5.)
    parser.add_argument("--range-uid", action="append")
    parser.add_argument("--max-ranges", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if any(not np.isfinite(x) for x in (args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes)):
        raise ValueError("Derivative lags must be finite")
    lag_map = lags(args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes)
    if args.max_ranges is not None and args.max_ranges <= 0:
        raise ValueError("max-ranges must be positive")
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    key = config_key(args.difference_method, args.dst_lag_hours, args.ae_lag_minutes, args.symh_lag_minutes, args.al_lag_minutes)
    output = args.output_dir or root / "environment_inputs/geomag_derivatives" / key
    master, provenance = selection.load_prepared_inputs(root, period)
    selected = selection.select_events(phases.classify_phases(master), args.min_occupied_blocks)
    rows = selected.drop_duplicates("range_uid").sort_values("range_start")
    if args.range_uid:
        unknown = set(args.range_uid) - set(rows.range_uid)
        if unknown:
            raise ValueError(f"Unknown or unselected range_uid: {sorted(unknown)}")
        rows = rows.loc[rows.range_uid.isin(args.range_uid)]
    if args.max_ranges:
        rows = rows.iloc[:args.max_ranges]
    if rows.empty:
        raise ValueError("No selected ranges")
    paths, missing = file_inventory(args.spedas_dir, rows, lag_map, args.difference_method)
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"method": METHOD, "difference_method": args.difference_method,
                     "dst_lag_hours": args.dst_lag_hours, "ae_lag_minutes": args.ae_lag_minutes,
                     "symh_lag_minutes": args.symh_lag_minutes, "al_lag_minutes": args.al_lag_minutes,
                     "unit": "nT/h", "smoothing": "none",
                     "stencil_gap_policy": "all native slots including endpoints must be finite",
                     "range_alignment": "nearest native derivative; half-cadence tolerance; never drop invalid slots"}
    plan = {"configuration": configuration, "range_uids": rows.range_uid.tolist(),
            "native_files": {k: [str(p) for p in v] for k, v in paths.items()},
            "missing_file_units": missing, "network": "none; local CDF reads only"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Planned {len(rows)} ranges: {output}; missing CDF units: {missing}", flush=True)
    if args.plan_only:
        return 0
    hashes = {p: batch.file_sha256(p) for p in {p for files in paths.values() for p in files}}
    sources = {k: [{"path": str(p), "sha256": hashes[p]} for p in v] for k, v in paths.items()}
    signature_base = {"configuration": configuration, "sources": sources,
                      "master_sha256": provenance["annual_master_sha256"],
                      "code_sha256": batch.file_sha256(Path(__file__)),
                      "summary_code_sha256": hashlib.sha256(inspect.getsource(describe).encode()).hexdigest()}
    rates = None
    cache = output / "ranges"
    cache.mkdir(exist_ok=True)
    results = []
    failures = []
    for _, row in rows.iterrows():
        uid = row.range_uid
        summary = Path(row.source_summary)
        timeseries = summary.with_name(summary.name.replace("kappa_phase_summary_", "kappa_timeseries_"))
        try:
            signature = hashlib.sha256(json.dumps({**signature_base, "range_uid": uid,
                "timeseries_sha256": batch.file_sha256(timeseries)}, sort_keys=True).encode()).hexdigest()
            result_path = cache / (hashlib.sha256(uid.encode()).hexdigest()[:20] + ".json")
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            if result.get("signature") != signature or result.get("status") != "complete":
                if rates is None:
                    rates = {k: differentiate(read_native(paths[k], spec["variable"]), spec["cadence"],
                                             lag_map[k], args.difference_method) for k, spec in SPECS.items()}
                time = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(timeseries, usecols=["time"])["time"]))
                time = time[(time >= pd.Timestamp(row.range_start)) & (time <= pd.Timestamp(row.range_end))].sort_values()
                result = {"range_uid": uid, "signature": signature, "status": "complete", "method": METHOD}
                for parameter, spec in SPECS.items():
                    reference = "range_geomag_" + spec["index"] + "_nT_n_total"
                    if len(time) != int(row[reference]):
                        raise ValueError(f"Saved grid sample count differs from master: {reference}")
                    values = sample_rates(rates[parameter], time, spec["cadence"])
                    result.update(describe(values, spec["prefix"]))
                tmp = result_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                tmp.replace(result_path)
            results.append(result)
            print(f"{uid}: complete", flush=True)
        except Exception as exc:
            failures.append({"range_uid": uid, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{uid}: failed: {exc}", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError("Derivative preparation incomplete; inspect failures.json")
    csv_path = output / "range_summary.csv"
    tmp = csv_path.with_suffix(".tmp")
    pd.DataFrame(results).to_csv(tmp, index=False)
    tmp.replace(csv_path)
    metadata = {**signature_base, "created_utc": datetime.now(timezone.utc).isoformat(), "method": METHOD,
                "period": period.key, "summary_sha256": batch.file_sha256(csv_path),
                "missing_file_units": missing, "n_ranges": len(results), "min_occupied_blocks": args.min_occupied_blocks,
                "warning": "current local CDF snapshot; original analysis CDF checksums unavailable"}
    csv_path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for parameter, spec in SPECS.items():
        count = sum((r[spec["prefix"] + "_valid_fraction"] >= .8) for r in results)
        print(f"{parameter}: {count}/{len(results)} ranges with valid fraction >= 0.8", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
