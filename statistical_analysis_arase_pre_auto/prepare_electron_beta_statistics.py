#!/usr/bin/env python3
"""Reconstruct electron beta only, without changing saved KAW fits or run state."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import batch_arase as batch
import statistics_phases as phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period


PREFIX = "range_electron_plasma_beta_HFA"
METHOD = "beta_e_hfa_v1_before_100s_smoothing_on_saved_kappa_time_grid"
REFERENCE_PREFIXES = ("range_ion_plasma_beta_HFA", "range_i_e_temp_ratio",
                      "range_number_density_HFA_cc", "range_temp_electron_eV")


def describe(values, prefix):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {prefix + "_n_total": len(values), prefix + "_n_valid": len(finite),
              prefix + "_valid_fraction": len(finite) / len(values) if len(values) else 0.}
    result.update({prefix + "_" + name: None for name in ("mean", "std", "median", "q16", "q84")})
    if len(finite):
        q16, median, q84 = np.percentile(finite, [16, 50, 84])
        result.update({prefix + "_mean": float(finite.mean()),
                       prefix + "_std": float(finite.std(ddof=1)) if len(finite) > 1 else None,
                       prefix + "_median": float(median), prefix + "_q16": float(q16),
                       prefix + "_q84": float(q84)})
    return result


def notebook_prefix(notebook, start, end, scratch, spedas_dir):
    """Use authoritative plasma construction, stopping before geomagnetic/CWT/fit cells."""
    nb = json.loads(Path(notebook).read_text())
    result = [batch.BATCH_PREAMBLE,
              "import matplotlib.figure\nmatplotlib.figure.Figure.savefig = lambda *a, **k: None\n"]
    parameterized = 0
    found = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        original = "".join(cell["source"])
        # Waveform diagnostics have no dependencies needed for plasma products.
        if "with ProcessPoolExecutor" in original and any(marker in original for marker in (
                "plot_EB_window_all_segs", "plot_fac_dsi_frame_polar")):
            continue
        if "da_hfa_spectral =" in original:
            continue
        # The current analysis uses observed ion velocity without subtracting
        # spacecraft velocity. These diagnostic cells are not plasma inputs.
        # In particular, their dsi2j2000 default noload=False reloads ORB/ATT.
        if any(marker in original for marker in (
                "def derive_spacecraft_velocity_gse(", "def transform_v_sc_to_dsi(",
                "v_sc_gse_analysis", "v_sc_j2000_analysis", "v_sc_dsi_analysis", "v_sc_fac_analysis")):
            continue
        source = batch.transform_notebook_source(original, start, end, scratch)
        if "def rotate_vector_dsi_to_fac(" in source:
            # Keep the function for the ion velocity, omit the v_sc call.
            tree = ast.parse(source)
            source = ast.unparse(ast.Module(body=[node for node in tree.body
                if isinstance(node, ast.FunctionDef)], type_ignores=[]))
        source = source.replace("print(v_sc_fac.time)", "")
        if "time_range_input =" in original:
            parameterized += 1
        if 'os.environ["SPEDAS_DATA_DIR"] =' in source:
            source = f"import os\nos.environ['SPEDAS_DATA_DIR'] = {str(spedas_dir)!r}"
        source = source.replace("PLOT_FAC_DSI_FRAME = True", "PLOT_FAC_DSI_FRAME = False")
        source = source.replace("PLOT_PRE_WAVELET_DIAGNOSTICS = True", "PLOT_PRE_WAVELET_DIAGNOSTICS = False")
        if "def build_plasma_parameter_products(" in source:
            marker = "    beta_hfa = (vthi / vA_hfa)**2"
            field = "        'ion_plasma_beta_HFA': sm(beta_hfa),"
            if source.count(marker) != 1 or source.count(field) != 1:
                raise ValueError("Plasma beta source contract changed")
            source = source.replace(marker, marker + "\n"
                                    "    beta_e_hfa = 2.0 * mu0 * ne_hfa * 1e6 * elementary_charge * te / (B_total * 1e-9)**2\n"
                                    "    beta_e_hfa = beta_e_hfa.where((ne_hfa > 0) & (te > 0) & (B_total > 0))")
            source = source.replace(field, field + "\n        'electron_plasma_beta_HFA': sm(beta_e_hfa),")
            found = True
        ast.parse(source)
        result.append(source)
        if found:
            break
    if not found or parameterized != 1:
        raise ValueError("Could not identify unique parameterized plasma prefix")
    combined = "\n".join(result)
    if "run_wavelet_segments(" in combined or "bootstrap_samples_path =" in combined:
        raise ValueError("Plasma prefix unexpectedly includes CWT/fit execution")
    if any(isinstance(node, ast.Name) and node.id.startswith("v_sc") for node in ast.walk(ast.parse(combined))):
        raise ValueError("Plasma prefix unexpectedly depends on spacecraft velocity")
    return result


def validate_reference(rebuilt, expected):
    for column, reference in expected.items():
        value = rebuilt[column]
        if reference is None and value is None:
            continue
        if reference is None or value is None or not np.isclose(value, reference, rtol=1e-6, atol=1e-10):
            raise ValueError(f"Reconstructed plasma context differs: {column}: {value} vs {reference}")


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    for field, path in (("preparer_sha256", Path(__file__)),
                        ("notebook_sha256", Path(request["notebook"])),
                        ("batch_runner_sha256", Path(batch.__file__))):
        if request[field] != batch.file_sha256(path):
            raise RuntimeError(f"Beta_e source changed after planning ({field}); stop and restart the parent command")
    # No network by default, including implicit listing/update calls in libraries.
    if not request["allow_network"]:
        import socket
        def offline(*args, **kwargs):
            raise RuntimeError("Offline beta_e reconstruction: network request blocked")
        socket.create_connection = offline
        socket.getaddrinfo = offline
        socket.socket.connect = offline
        socket.socket.connect_ex = offline
    sys.path.insert(0, str(batch.ROOT))
    namespace = {"__name__": "__main__"}
    with tempfile.TemporaryDirectory(prefix="arase_beta_e_") as temporary:
        sources = notebook_prefix(request["notebook"], request["start"], request["end"],
                                  Path(temporary), Path(request["spedas_dir"]))
        for index, source in enumerate(sources):
            exec(compile(source, f"beta_e_plasma_cell_{index}", "exec"), namespace)
        ds = namespace["ds_parameter"].sortby("time")
        # Match the original range_context grid rather than averaging native samples.
        times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(request["timeseries"], usecols=["time"])["time"]))
        times = times[(times >= pd.Timestamp(request["start"])) & (times <= pd.Timestamp(request["end"]))].sort_values()
        if times.empty or times.has_duplicates:
            raise ValueError("Saved kappa time grid is empty or duplicated")
        ds = ds.interp(time=times)
        rebuilt = {}
        for prefix in REFERENCE_PREFIXES:
            field = prefix.removeprefix("range_")
            if field == "i_e_temp_ratio":
                field = "i-e_temp_ratio"
            rebuilt.update(describe(ds[field].values, prefix))
        validate_reference(rebuilt, request["reference"])
        result = describe(ds["electron_plasma_beta_HFA"].values, PREFIX)
        if not result[PREFIX + "_n_valid"]:
            raise ValueError("No finite reconstructed electron beta samples")
    result.update({"range_uid": request["range_uid"], "signature": request["signature"],
                   "method": METHOD, "status": "complete", "reference_check": "passed"})
    target = Path(request["result"])
    temporary_path = target.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(target)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--notebook", type=Path, default=batch.DEFAULT_NOTEBOOK)
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--range-uid", action="append")
    parser.add_argument("--max-ranges", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true", help="Permit loaders to acquire missing cached files")
    parser.add_argument("--timeout", type=int, default=14400, help="Seconds per isolated range")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or (args.max_ranges is not None and args.max_ranges <= 0):
        raise ValueError("timeout and max-ranges must be positive")
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / "environment_inputs/beta_e"
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
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "ranges"
    cache.mkdir(exist_ok=True)
    notebook_hash = batch.file_sha256(args.notebook)
    code_hash = batch.file_sha256(Path(__file__))
    runner_hash = batch.file_sha256(Path(batch.__file__))
    requests = []
    for _, row in rows.iterrows():
        summary = Path(row.source_summary)
        timeseries = summary.with_name(summary.name.replace("kappa_phase_summary_", "kappa_timeseries_"))
        if not timeseries.is_file():
            raise FileNotFoundError(timeseries)
        reference = {prefix + "_" + field: (float(row[prefix + "_" + field])
                     if pd.notna(row[prefix + "_" + field]) else None)
                     for prefix in REFERENCE_PREFIXES for field in ("n_total", "n_valid", "median", "q16", "q84")}
        uid_hash = hashlib.sha256(row.range_uid.encode()).hexdigest()[:20]
        request = {"range_uid": row.range_uid, "start": row.range_start, "end": row.range_end,
                   "notebook": str(args.notebook.resolve()), "notebook_sha256": notebook_hash,
                   "preparer_sha256": code_hash, "batch_runner_sha256": runner_hash,
                   "timeseries": str(timeseries), "timeseries_sha256": batch.file_sha256(timeseries),
                   "reference": reference, "master_sha256": provenance["annual_master_sha256"],
                   "spedas_dir": str(args.spedas_dir.resolve()), "allow_network": args.allow_network,
                   "result": str((cache / f"{uid_hash}.json").resolve()), "method": METHOD}
        request["signature"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        requests.append(request)
    # Validate the extraction structure without reading CDFs or executing source cells.
    if len(rows):
        notebook_prefix(args.notebook, rows.iloc[0].range_start, rows.iloc[0].range_end,
                        Path("/tmp/arase_beta_e_plan"), args.spedas_dir)
    (output / "plan.json").write_text(json.dumps(requests, indent=2) + "\n")
    print(f"Planned {len(requests)} unique ranges; offline={not args.allow_network}: {output}", flush=True)
    if args.plan_only:
        return 0
    results, failures = [], []
    for index, request in enumerate(requests):
        result_path = Path(request["result"])
        cached = json.loads(result_path.read_text()) if result_path.is_file() else {}
        if cached.get("signature") != request["signature"] or cached.get("status") != "complete":
            request_path = result_path.with_suffix(".request.json")
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            log_path = result_path.with_suffix(".log")
            env = dict(os.environ, MPLBACKEND="Agg", SPEDAS_DATA_DIR=str(args.spedas_dir),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request_path)]
            log_path.write_text("")
            code = batch.run_child(command, log_path, args.timeout, env)
            if code != 0 or not result_path.is_file():
                failures.append({"range_uid": request["range_uid"], "returncode": code, "log": str(log_path)})
                (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
                print(f"Failed {request['range_uid']}: {log_path}", flush=True)
                log_text = log_path.read_text(errors="replace")
                if ("Offline beta_e reconstruction: network request blocked" in log_text
                        or "Beta_e source changed after planning" in log_text):
                    raise RuntimeError(f"Beta_e preparation stopped at first offline/source-contract failure; inspect {log_path}")
                continue
            cached = json.loads(result_path.read_text())
        if cached.get("signature") != request["signature"] or cached.get("status") != "complete":
            raise ValueError(f"Invalid result: {result_path}")
        results.append(cached)
        print(f"[{index+1}/{len(requests)}] {request['range_uid']}: complete", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures or not results:
        raise RuntimeError("Electron beta preparation incomplete; see failures.json and per-range logs")
    frame = pd.DataFrame(results)
    csv_path = output / "range_summary.csv"
    temporary_csv = csv_path.with_suffix(".tmp")
    frame.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "method": METHOD,
                "period": period.key, "master_sha256": provenance["annual_master_sha256"],
                "summary_sha256": batch.file_sha256(csv_path), "n_ranges": len(frame),
                "notebook_sha256": notebook_hash, "preparer_sha256": code_hash,
                "batch_runner_sha256": runner_hash, "offline": not args.allow_network,
                "min_occupied_blocks": args.min_occupied_blocks,
                "formula": "smooth100s(2*mu0*ne_HFA[m^-3]*e*Te[eV]/B[T]^2), then saved kappa grid interpolation and range median",
                "cdf_provenance": "Uses current local CDF cache; original CDF checksums unavailable; reconstructed ion beta, Ti/Te, ne, Te quantiles checked against saved master"}
    csv_path.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-request":
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
