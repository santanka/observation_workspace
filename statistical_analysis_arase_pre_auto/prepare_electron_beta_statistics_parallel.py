#!/usr/bin/env python3
"""Parallel scheduling of the unchanged, isolated electron-beta worker."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import prepare_electron_beta_statistics as serial


def positive_workers(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("--workers must be at least 1")
    return value


def build_parser():
    parser = serial.build_parser()
    parser.description = __doc__
    parser.add_argument("--workers", type=positive_workers, default=1,
                        help="Concurrent isolated beta_e range processes (default: 1)")
    return parser


def serial_arguments(args):
    result = []
    for name, value in vars(args).items():
        if name == "workers" or value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            result.append(flag)
        else:
            for item in value if isinstance(value, list) else [value]:
                result.extend([flag, str(item)])
    return result


def read_complete(request):
    path = Path(request["result"])
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result.get("signature") == request["signature"] and result.get("status") == "complete":
        return result
    return None


def worker_command(request_path):
    return [sys.executable, str(Path(serial.__file__).resolve()), "--worker-request", str(request_path)]


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def legacy_processes(output, proc_root=Path("/proc")):
    """Detect a legacy parent or orphan worker writing this same output."""
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = [os.fsdecode(a) for a in (process / "cmdline").read_bytes().split(b"\0") if a]
            script_index = next((i for i, a in enumerate(arguments)
                                 if Path(a).name == Path(serial.__file__).name), None)
            if script_index is None:
                continue
            options = arguments[script_index+1:]
            if "--worker-request" in options:
                request = Path(options[options.index("--worker-request")+1])
                same_output = request.resolve().parent == (output / "ranges").resolve()
            else:
                with contextlib.redirect_stderr(io.StringIO()):
                    args, _ = serial.build_parser().parse_known_args(options)
                period = serial.resolve_period(args)
                root = args.statistics_dir or serial.batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
                same_output = (args.output_dir or root / "environment_inputs/beta_e").resolve() == output.resolve()
            if same_output:
                found.append(int(process.name))
        except (OSError, ValueError, IndexError, SystemExit):
            continue  # Processes may exit while being inspected.
    return found


class RangePool:
    def __init__(self, timeout, spedas_dir):
        self.timeout = timeout
        self.environment = dict(os.environ, MPLBACKEND="Agg", SPEDAS_DATA_DIR=str(spedas_dir),
                                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.processes = {}

    def terminate(self):
        self.stop.set()
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            serial.batch.terminate_process_group(process)

    def execute(self, request):
        result_path = Path(request["result"])
        request_path = result_path.with_suffix(".request.json")
        log_path = result_path.with_suffix(".log")
        if self.stop.is_set():
            return {"range_uid": request["range_uid"], "status": "cancelled"}
        atomic_json(request_path, request)
        command = worker_command(request_path)
        with log_path.open("w", buffering=1) as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] command: {' '.join(command)}\n")
            # Register under the same lock used by cancellation so no process escapes it.
            with self.lock:
                if self.stop.is_set():
                    return {"range_uid": request["range_uid"], "status": "cancelled"}
                process = subprocess.Popen(command, cwd=serial.batch.ROOT, env=self.environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.processes[request["range_uid"]] = process
            try:
                try:
                    code = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    log.write(f"timeout after {self.timeout} seconds\n")
                    serial.batch.terminate_process_group(process)
                    code = 124
            finally:
                with self.lock:
                    self.processes.pop(request["range_uid"], None)
        if self.stop.is_set():
            return {"range_uid": request["range_uid"], "status": "cancelled"}
        if code == 0 and read_complete(request) is not None:
            return {"range_uid": request["range_uid"], "status": "complete"}
        tail = log_path.read_text(errors="replace")
        fatal = ("Offline beta_e reconstruction: network request blocked" in tail
                 or "Beta_e source changed after planning" in tail)
        return {"range_uid": request["range_uid"], "status": "failed", "returncode": code,
                "log": str(log_path), "fatal": fatal}


def run_requests(requests, workers, timeout, spedas_dir, output):
    if len({r["range_uid"] for r in requests}) != len(requests):
        raise ValueError("Duplicate range_uid in beta_e plan")
    cached = [r for r in requests if read_complete(r) is not None]
    cached_uids = {r["range_uid"] for r in cached}
    remaining = iter(r for r in requests if r["range_uid"] not in cached_uids)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers,
              "planned": len(requests), "cached_complete": len(cached), "new_complete": 0,
              "failed": [], "cancelled": [], "status": "running"}
    report_path = output / "parallel_execution.json"
    atomic_json(report_path, report)
    print(f"beta_e: workers={workers}, reused={len(cached)}, pending={len(requests)-len(cached)}", flush=True)
    pool = RangePool(timeout, spedas_dir)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    def submit():
        request = next(remaining, None)
        if request is not None:
            pending[executor.submit(pool.execute, request)] = request
    try:
        for _ in range(workers):
            submit()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            fatal = False
            for future in completed:
                request = pending.pop(future)
                outcome = future.result()
                if outcome["status"] == "complete":
                    report["new_complete"] += 1
                    print(f"[{len(cached)+report['new_complete']}/{len(requests)}] {request['range_uid']}: complete", flush=True)
                elif outcome["status"] == "failed":
                    report["failed"].append(outcome)
                    fatal |= outcome["fatal"]
                    atomic_json(output / "failures.json", report["failed"])
                    print(f"Failed {request['range_uid']}: {outcome['log']}", flush=True)
                else:
                    report["cancelled"].append(outcome["range_uid"])
            atomic_json(report_path, report)
            if fatal:
                raise RuntimeError("Offline/source-contract failure; beta_e parallel scheduling stopped")
            for _ in completed:
                submit()
        if report["failed"]:
            raise RuntimeError("Electron beta preparation incomplete; inspect failures.json")
        report["status"] = "complete"
        atomic_json(output / "failures.json", [])
    except BaseException as exc:
        pool.terminate()
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["cancelled"].extend(r["range_uid"] for r in pending.values())
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or (args.max_ranges is not None and args.max_ranges <= 0):
        raise ValueError("timeout and max-ranges must be positive")
    period = serial.resolve_period(args)
    root = args.statistics_dir or serial.batch.DEFAULT_OUTPUT_ROOT / "statistics" / period.key
    output = args.output_dir or root / "environment_inputs/beta_e"
    output.mkdir(parents=True, exist_ok=True)
    # The legacy serial command does not acquire this lock: stop it before switching.
    with (output / ".parallel.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another parallel beta_e coordinator holds this output directory") from exc
        active = legacy_processes(output)
        if active:
            raise RuntimeError(f"Legacy beta_e parent/worker still uses this output (PIDs {active}); stop the old job and wait for its workers to finish")
        options = serial_arguments(args)
        if args.workers == 1 or args.plan_only:
            return serial.main(options)
        # Keep the existing worker, request signatures, reference checks and final
        # aggregation unchanged so successful serial results remain reusable.
        serial.main([*options, "--plan-only"])
        requests = json.loads((output / "plan.json").read_text())
        run_requests(requests, args.workers, args.timeout, args.spedas_dir, output)
        # Reuse the authoritative aggregation and checksum verification, but do
        # not silently launch serial work if inputs changed during execution.
        original_run_child = serial.batch.run_child
        def reject_new_work(*_args, **_kwargs):
            raise RuntimeError("Inputs changed after the parallel plan; restart the parallel command")
        try:
            serial.batch.run_child = reject_new_work
            serial.main(options)
        finally:
            serial.batch.run_child = original_run_child
        provenance_path = output / "range_summary.provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["parallel_execution"] = {"workers": args.workers,
            "coordinator_sha256": serial.batch.file_sha256(Path(__file__)),
            "report": str(output / "parallel_execution.json")}
        atomic_json(provenance_path, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
