#!/usr/bin/env python3
"""Resumable isolated FAC Vy reconstruction; preserve original KAW fits and beta_e cache."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pandas as pd

import batch_arase as batch
import prepare_electron_beta_statistics as plasma
from prepare_electron_beta_statistics_parallel import atomic_json, positive_workers
import statistics_phases as phases
import statistics_selection as selection
from statistics_period import add_period_arguments, resolve_period


PREFIX = "range_sys_velocity_fac_y_ms"
X_PREFIX = "range_perp_sys_speed_ms"
METHOD = "fac_vsys_y_100s_on_saved_kappa_grid_v1"
FIELDS = ("n_total", "n_valid", "valid_fraction", "mean", "std", "median", "q16", "q84")


def source_paths(request):
    return {"preparer_sha256": Path(__file__), "plasma_preparer_sha256": Path(plasma.__file__),
            "notebook_sha256": Path(request["notebook"]), "batch_runner_sha256": Path(batch.__file__),
            "timeseries_sha256": Path(request["timeseries"])}


def validate_sources(request, fields=None):
    for field, path in source_paths(request).items():
        if fields is not None and field not in fields:
            continue
        if batch.file_sha256(path) != request[field]:
            raise RuntimeError(f"Vy source changed after planning: {field}")


@contextmanager
def offline_network(enabled):
    def blocked(*_args, **_kwargs):
        raise RuntimeError("Offline Vy reconstruction: network request blocked")
    targets = ((socket, "create_connection"), (socket, "getaddrinfo"),
               (socket.socket, "connect"), (socket.socket, "connect_ex"))
    originals = [(target, name, getattr(target, name)) for target, name in targets]
    try:
        if enabled:
            for target, name, _ in originals:
                setattr(target, name, blocked)
        yield
    finally:
        if enabled:
            for target, name, original in originals:
                setattr(target, name, original)


def prefix_sources(request, scratch):
    sources = plasma.notebook_prefix(request["notebook"], request["start"], request["end"],
                                    scratch, Path(request["spedas_dir"]))
    combined = "\n".join(sources)
    for marker in ("vsys_y = vector_component(vsys_fac, index=1, label='y_FAC')",
                   "'perp_sys_speed': sm(vsys_y)", "v_sys_fac = v_ion_fac.rename('v_sys_fac')",
                   "build_plasma_parameter_products(time_base, smoothing_sec=100.0)"):
        if marker not in combined:
            raise ValueError(f"Vy plasma source contract changed: {marker}")
    return sources


def execute_prefix(request, scratch):
    namespace = {"__name__": "__main__"}
    for index, source in enumerate(prefix_sources(request, scratch)):
        exec(compile(source, f"vsys_y_plasma_cell_{index}", "exec"), namespace)
    return namespace


def summarize_velocity(namespace, times, reference):
    """Use the same 100-s smoothed products and time grid as the saved Vx summary."""
    rebuilt = {}
    y_result = None
    for component, dataset, prefix in (("x", "ds_velocity_ms_toroidal", X_PREFIX),
                                        ("y", "ds_velocity_ms_poloidal", PREFIX)):
        ds = namespace[dataset].sortby("time")
        _, keep = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=np.sort(keep)).interp(time=times)
        stats = plasma.describe(ds["perp_sys_speed"].values, prefix)
        if component == "x":
            rebuilt = stats
            for field in ("n_total", "n_valid"):
                name = X_PREFIX + "_" + field
                if stats[name] != reference[name]:
                    raise ValueError(f"Reconstructed Vx sample count differs: {name}")
            plasma.validate_reference(stats, reference)
        else:
            y_result = stats
    if not y_result[PREFIX + "_n_valid"]:
        raise ValueError("No finite reconstructed Vy samples")
    return y_result, rebuilt


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    result_path = Path(request["result"])
    with result_path.with_suffix(".worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Vy worker holds this range") from exc
        validate_sources(request)
        # Notebook cells import workspace modules; a script invocation only puts
        # this subdirectory on sys.path, even when its cwd is the workspace root.
        sys.path.insert(0, str(batch.ROOT))
        with offline_network(not request["allow_network"]), tempfile.TemporaryDirectory(prefix="arase_vsys_y_") as temporary:
            namespace = execute_prefix(request, Path(temporary))
            times = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(request["timeseries"], usecols=["time"])["time"], utc=True)).tz_convert(None)
            start = pd.Timestamp(request["start"]).tz_localize(None)
            end = pd.Timestamp(request["end"]).tz_localize(None)
            times = times[(times >= start) & (times <= end)].sort_values()
            if times.empty or times.has_duplicates:
                raise ValueError("Saved kappa time grid is empty or duplicated")
            result, reference = summarize_velocity(namespace, times, request["reference"])
        validate_sources(request)
        result.update({"range_uid": request["range_uid"], "signature": request["signature"],
                       "method": METHOD, "status": "complete", "reference_check": "passed",
                       "reconstructed_vx": reference})
        atomic_json(result_path, result)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_period_arguments(parser)
    selection.add_selection_argument(parser)
    parser.add_argument("--statistics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--notebook", type=Path, default=batch.DEFAULT_NOTEBOOK)
    parser.add_argument("--spedas-dir", type=Path, default=batch.DEFAULT_SPEDAS_DIR)
    parser.add_argument("--workers", type=positive_workers, default=2)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--range-uid", action="append")
    parser.add_argument("--max-ranges", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser


def build_requests(args, master, provenance, output):
    selected = selection.select_events(phases.classify_phases(master), args.min_occupied_blocks)
    reference_columns = [X_PREFIX + "_" + f for f in FIELDS]
    required = ["range_start", "range_end", "source_summary", *reference_columns]
    if not selected["system_velocity_policy"].eq("observed_partial_ion_moment_no_spacecraft_subtraction").all():
        raise ValueError("Vy requires the observed-ion v2 system velocity policy")
    if not selected.source_summary.str.contains("/EB_ratio_toroidal_eachtime/", regex=False).all():
        raise ValueError("Vx reference must come from the saved toroidal analysis")
    if (selected.groupby("range_uid")[required].nunique(dropna=False) > 1).any().any():
        raise ValueError("Inconsistent range context across phases")
    rows = selected.drop_duplicates("range_uid").sort_values("range_start")
    if args.range_uid:
        if set(args.range_uid) - set(rows.range_uid):
            raise ValueError("Unknown or unselected range_uid")
        rows = rows.loc[rows.range_uid.isin(args.range_uid)]
    if args.max_ranges:
        rows = rows.iloc[:args.max_ranges]
    requests = []
    # Hash shared source files once per plan, not once for every range.
    common_paths = {"preparer_sha256": Path(__file__), "plasma_preparer_sha256": Path(plasma.__file__),
                    "notebook_sha256": args.notebook, "batch_runner_sha256": Path(batch.__file__)}
    common_hashes = {field: batch.file_sha256(path) for field, path in common_paths.items()}
    for _, row in rows.iterrows():
        summary = Path(row.source_summary)
        timeseries = summary.with_name(summary.name.replace("kappa_phase_summary_", "kappa_timeseries_"))
        request = {"range_uid": row.range_uid, "start": row.range_start, "end": row.range_end,
                   "notebook": str(args.notebook.resolve()), "timeseries": str(timeseries),
                   "spedas_dir": str(args.spedas_dir.resolve()), "allow_network": args.allow_network,
                   "master_sha256": provenance["annual_master_sha256"], "method": METHOD,
                   "reference": {c: float(row[c]) if pd.notna(row[c]) else None for c in reference_columns},
                   "result": str((output / "ranges" / (hashlib.sha256(row.range_uid.encode()).hexdigest()[:20] + ".json")).resolve())}
        request.update(common_hashes)
        request["timeseries_sha256"] = batch.file_sha256(timeseries)
        # Download permission does not change the scientific values of an already verified range.
        signature_fields = {k: v for k, v in request.items() if k != "allow_network"}
        request["signature"] = hashlib.sha256(json.dumps(signature_fields, sort_keys=True).encode()).hexdigest()
        requests.append(request)
    if not requests:
        raise ValueError("No selected Vy ranges")
    prefix_sources(requests[0], Path("/tmp/arase_vsys_y_plan"))  # AST/contract only; no CDF execution.
    return requests


def read_complete(request):
    path = Path(request["result"])
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if (result.get("signature") == request["signature"] and result.get("status") == "complete"
            and result.get("method") == METHOD and result.get("reference_check") == "passed"
            and result.get("range_uid") == request["range_uid"]
            and all(PREFIX + "_" + f in result for f in FIELDS)
            and result.get(PREFIX + "_n_valid", 0) > 0):
        return result
    return None


def check_worker_locks(output):
    for path in (output / "ranges").glob("*.worker.lock"):
        with path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"An orphan Vy worker still holds {path}; wait for it to finish") from exc


class RangePool:
    def __init__(self, args, coordinator_lock_fd=None):
        self.args = args
        self.coordinator_lock_fd = coordinator_lock_fd
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.processes = {}

    def terminate(self):
        self.stop.set()
        with self.lock:
            for process in self.processes.values():
                batch.terminate_process_group(process)

    def execute(self, request):
        path = Path(request["result"])
        request_path, log_path = path.with_suffix(".request.json"), path.with_suffix(".log")
        atomic_json(request_path, request)
        environment = dict(os.environ, MPLBACKEND="Agg", SPEDAS_DATA_DIR=str(self.args.spedas_dir),
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with log_path.open("w", buffering=1) as log:
            with self.lock:
                if self.stop.is_set():
                    return {"range_uid": request["range_uid"], "status": "cancelled"}
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request_path)],
                    cwd=batch.ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=(() if self.coordinator_lock_fd is None else (self.coordinator_lock_fd,)))
                self.processes[request["range_uid"]] = process
            try:
                try:
                    code = process.wait(timeout=self.args.timeout)
                except subprocess.TimeoutExpired:
                    batch.terminate_process_group(process)
                    code = 124
            except BaseException:
                batch.terminate_process_group(process)
                raise
            finally:
                with self.lock:
                    self.processes.pop(request["range_uid"], None)
        if code == 0 and read_complete(request) is not None:
            return {"range_uid": request["range_uid"], "status": "complete"}
        text = log_path.read_text(errors="replace")
        fatal = any(s in text for s in ("Offline Vy reconstruction", "Vy source changed", "source contract changed",
                                       "Reconstructed plasma context differs", "Reconstructed Vx sample count differs"))
        return {"range_uid": request["range_uid"], "status": "failed", "returncode": code,
                "log": str(log_path), "fatal": fatal}


def run_requests(requests, args, output, coordinator_lock_fd=None):
    pending_requests = [r for r in requests if read_complete(r) is None]
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "workers": args.workers,
              "planned": len(requests), "cached_complete": len(requests) - len(pending_requests),
              "new_complete": 0, "failed": []}
    atomic_json(output / "execution.json", report)
    print(f"Vy: reused={report['cached_complete']}, pending={len(pending_requests)}, workers={args.workers}", flush=True)
    pool = RangePool(args, coordinator_lock_fd)
    executor = ThreadPoolExecutor(max_workers=args.workers)
    pending, remaining = {}, iter(pending_requests)
    def submit():
        request = next(remaining, None)
        if request is not None:
            pending[executor.submit(pool.execute, request)] = request
    try:
        for _ in range(args.workers):
            submit()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            fatal = False
            for future in completed:
                request = pending.pop(future)
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = {"range_uid": request["range_uid"], "status": "failed", "fatal": True,
                               "error": f"{type(exc).__name__}: {exc}",
                               "log": str(Path(request["result"]).with_suffix(".log"))}
                if outcome["status"] == "complete":
                    report["new_complete"] += 1
                    print(f"[{report['cached_complete'] + report['new_complete']}/{len(requests)}] {request['range_uid']}: complete", flush=True)
                else:
                    report["failed"].append(outcome)
                    fatal |= outcome.get("fatal", False)
                    atomic_json(output / "failures.json", report["failed"])
                    print(f"Failed {request['range_uid']}: {outcome.get('log')}", flush=True)
            atomic_json(output / "execution.json", report)
            if fatal:
                raise RuntimeError("Vy preparation stopped after offline/source/reference failure; inspect per-range log")
            for _ in completed:
                submit()
        if report["failed"]:
            raise RuntimeError("Vy preparation incomplete; rerun to retry failed ranges")
        report["status"] = "complete"
        atomic_json(output / "failures.json", [])
    except BaseException as exc:
        pool.terminate()
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "execution.json", report)


def aggregate(requests, args, period, provenance, output):
    results = []
    for request in requests:
        result = read_complete(request)
        if result is None:
            raise ValueError("Vy summary requires every planned range to be complete")
        if not results:
            validate_sources(request)
        else:
            for field in ("preparer_sha256", "plasma_preparer_sha256", "notebook_sha256", "batch_runner_sha256"):
                if request[field] != requests[0][field]:
                    raise ValueError("Mixed scientific source signatures in Vy plan")
            validate_sources(request, fields=("timeseries_sha256",))
        # Detailed reference values stay in per-range JSON; CSV contains the Y statistics.
        results.append({k: v for k, v in result.items() if k != "reconstructed_vx"})
    validate_sources(requests[0])
    master_path = (args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key) / "inputs/arase_phase_master.csv"
    if batch.file_sha256(master_path) != provenance["annual_master_sha256"]:
        raise ValueError("Prepared master changed during Vy processing")
    target = output / "range_summary.csv"
    temporary = target.with_suffix(".tmp")
    pd.DataFrame(results).to_csv(temporary, index=False)
    temporary.replace(target)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "method": METHOD, "period": period.key,
                "master_sha256": provenance["annual_master_sha256"], "summary_sha256": batch.file_sha256(target),
                "n_ranges": len(results), "min_occupied_blocks": args.min_occupied_blocks,
                "formula": "FAC y observed ion velocity -> original 100s smoothing -> saved kappa time grid -> signed range statistics",
                "reference": "FAC x n_total/n_valid exact; all other saved Vx summary fields rtol=1e-6, atol=1e-10",
                "units": "m/s", "offline": not args.allow_network,
                "code_sha256": {field: requests[0][field] for field in source_paths(requests[0]) if field != "timeseries_sha256"},
                "cdf_provenance": "current local CDF cache; original CDF hashes unavailable; saved Vx statistics verified",
                "plan_sha256": batch.file_sha256(output / "plan.json")}
    atomic_json(target.with_suffix(".provenance.json"), metadata)
    print(f"Wrote {target}", flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or (args.max_ranges is not None and args.max_ranges <= 0):
        raise ValueError("timeout and max-ranges must be positive")
    period = resolve_period(args)
    root = args.statistics_dir or batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / "environment_inputs/vsys_y"
    output.mkdir(parents=True, exist_ok=True)
    (output / "ranges").mkdir(exist_ok=True)
    with (output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Vy coordinator holds this output") from exc
        check_worker_locks(output)
        master, provenance = selection.load_prepared_inputs(root, period)
        requests = build_requests(args, master, provenance, output)
        atomic_json(output / "plan.json", requests)
        print(f"Planned {len(requests)} Vy ranges; offline={not args.allow_network}", flush=True)
        if args.plan_only:
            return 0
        run_requests(requests, args, output, coordinator_lock_fd=lock.fileno())
        aggregate(requests, args, period, provenance, output)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-request":
        worker(sys.argv[2])
    else:
        raise SystemExit(main())
