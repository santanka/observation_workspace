#!/usr/bin/env python3
"""Run Arase_event_preanalysis.ipynb in isolated monthly kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = Path(__file__).with_name("Arase_event_preanalysis.ipynb")
DEFAULT_START = "20200401/00:00:00"
DEFAULT_END = "20230101/00:00:00"
DEFAULT_STATE_DIR = Path(
    "/mnt/j/observation_data/statistical_analysis_arase/monthly_preanalysis_state"
)
DEFAULT_VALID_RANGES_DIR = Path(
    "/mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges"
)
DEFAULT_SPEDAS_DIR = Path("/mnt/j/observation_data")
DEFAULT_TIMEOUT_SEC = 24 * 3600
NO_DATA_EXIT_CODE = 20
NO_DATA_EXCEPTION_NAME = "NoUsableScienceData"


@dataclass(frozen=True)
class MonthRange:
    key: str
    start: datetime
    end: datetime

    @property
    def start_text(self) -> str:
        return self.start.strftime("%Y%m%d/%H:%M:%S")

    @property
    def end_text(self) -> str:
        return self.end.strftime("%Y%m%d/%H:%M:%S")


def parse_time(value: str) -> datetime:
    for fmt in ("%Y%m%d/%H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported time format: {value}")


def next_month(value: datetime) -> datetime:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1)
    return value.replace(month=value.month + 1)


def build_month_ranges(start_text: str, end_text: str) -> list[MonthRange]:
    start = parse_time(start_text)
    end = parse_time(end_text)
    for label, value in (("start", start), ("end", end)):
        if (value.day, value.hour, value.minute, value.second, value.microsecond) != (
            1, 0, 0, 0, 0
        ):
            raise ValueError(f"{label} must be a month boundary at 00:00:00")
    if start >= end:
        raise ValueError("start must be earlier than end")

    ranges = []
    cursor = start
    while cursor < end:
        month_end = next_month(cursor)
        if month_end > end:
            raise ValueError("end must be a month boundary reachable from start")
        ranges.append(MonthRange(cursor.strftime("%Y%m"), cursor, month_end))
        cursor = month_end
    return ranges


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def transform_notebook_source(
    source: str, month: MonthRange, valid_ranges_dir: Path | None = None
) -> tuple[str, int]:
    replacement = f"time_range = [{month.start_text!r}, {month.end_text!r}]"
    source, count = re.subn(
        r"^time_range\s*=\s*\[[^\n]+\]$",
        replacement,
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if valid_ranges_dir is not None:
        source = source.replace(
            f'Path("{DEFAULT_VALID_RANGES_DIR}")',
            f"Path({str(valid_ranges_dir)!r})",
        )
    return source, count


def expected_result(valid_ranges_dir: Path, month: MonthRange) -> Path:
    start = month.start.strftime("%Y%m%d_%H%M%S")
    end = month.end.strftime("%Y%m%d_%H%M%S")
    return valid_ranges_dir / f"Arase_valid_time_ranges_{start}_to_{end}.json"


def extract_no_data_reason(error: BaseException) -> str | None:
    """Notebookの専用例外から短いno_data理由を取り出す。"""
    plain_error = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(error))
    marker = f"{NO_DATA_EXCEPTION_NAME}:"
    for line in reversed(plain_error.splitlines()):
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return None


def no_data_result_payload(month: MonthRange, reason: str) -> dict[str, Any]:
    """必須データ全期間欠測時の空range JSONを作る。"""
    return {
        "description": (
            "No valid analysis ranges: a required Arase data product was "
            "unavailable for the full analysis interval."
        ),
        "source_notebook": DEFAULT_NOTEBOOK.name,
        "analysis_status": "no_data",
        "no_data_reason": reason,
        "time_range_input": [month.start_text, month.end_text],
        "event_half_width_sec": 200.0,
        "minimum_event_duration_sec": 400.0,
        "maximum_analysis_duration_sec": 1200.0,
        "background_context_pad_sec": 60.0,
        "range_boundary_epsilon_sec": 1e-6,
        "range_interval_convention": (
            "closed intervals; non-final segment ends 1 microsecond before "
            "the next segment starts"
        ),
        "n_parent_ranges": 0,
        "n_analysis_ranges": 0,
        "n_valid_ranges": 0,
        "n_split_parent_ranges": 0,
        "total_parent_duration_min": 0.0,
        "total_analysis_duration_min": 0.0,
        "total_duration_min": 0.0,
        "selection_summary": {
            "n_selected_times": 0,
            "n_selected_times_mono": 0,
            "n_forbidden_intervals_mono": 1,
        },
        "conditions": [
            "Required data product unavailable for the full analysis interval",
            "The full interval was excluded; absence was not treated as a non-detection",
        ],
        "unavailable_ranges": [{
            "start_time": month.start.isoformat(),
            "end_time": month.end.isoformat(),
            "reason": reason,
        }],
        "parent_ranges": [],
        "ranges": [],
    }


def status_path(state_dir: Path, month: MonthRange) -> Path:
    return state_dir / "months" / f"{month.key}.json"


def compatible_complete(
    status: dict[str, Any], notebook_hash: str, runner_hash: str, result: Path
) -> bool:
    return (
        status.get("status") in {"complete", "no_data"}
        and status.get("notebook_sha256") == notebook_hash
        and status.get("runner_sha256") == runner_hash
        and Path(status.get("result_path", "")) == result
        and result.is_file()
    )


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_child(command: list[str], log_path: Path, timeout: int, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1, encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat()}] command: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"timeout after {timeout} s\n")
            terminate_process_group(process)
            return 124


def execute_one(args: argparse.Namespace) -> int:
    import nbformat
    from nbclient import NotebookClient

    notebook_path = Path(args.notebook).resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    month = MonthRange(args.month, parse_time(args.start), parse_time(args.end))

    replacements = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.source, count = transform_notebook_source(
            cell.source, month, Path(args.valid_ranges_dir)
        )
        replacements += count
        cell.outputs = []
        cell.execution_count = None
    if replacements != 1:
        raise RuntimeError(f"Expected one time_range assignment, found {replacements}")

    failed_path = Path(args.failed_notebook)

    def on_cell_start(*, cell: Any, cell_index: int) -> None:
        first_line = cell.source.splitlines()[0][:100] if cell.source else ""
        print(f"cell {cell_index + 1}/{len(notebook.cells)} start: {first_line}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel_name,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
        on_cell_start=on_cell_start,
    )
    try:
        client.execute()
    except Exception as err:
        no_data_reason = extract_no_data_reason(err)
        if no_data_reason is not None:
            result = expected_result(Path(args.valid_ranges_dir), month)
            atomic_json(result, no_data_result_payload(month, no_data_reason))
            print(f"NO_DATA: {no_data_reason}", flush=True)
            print(f"empty result saved: {result}", flush=True)
            return NO_DATA_EXIT_CODE
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, failed_path)
        print(f"partial failed notebook saved: {failed_path}", flush=True)
        traceback.print_exc()
        return 1
    return 0


def run_months(args: argparse.Namespace) -> int:
    notebook = Path(args.notebook).resolve()
    state_dir = Path(args.state_dir)
    valid_ranges_dir = Path(args.valid_ranges_dir)
    ranges = build_month_ranges(args.start, args.end)
    if args.month:
        wanted = set(args.month)
        ranges = [month for month in ranges if month.key in wanted]
        missing = wanted - {month.key for month in ranges}
        if missing:
            raise ValueError(f"Requested months outside interval: {sorted(missing)}")

    print(f"monthly ranges : {len(ranges)}")
    print(f"first month    : {ranges[0].start_text} -- {ranges[0].end_text}")
    print(f"last month     : {ranges[-1].start_text} -- {ranges[-1].end_text}")
    if args.dry_run:
        for month in ranges:
            print(month.key, month.start_text, month.end_text)
        return 0

    notebook_hash = file_sha256(notebook)
    runner_hash = file_sha256(Path(__file__).resolve())
    env = os.environ.copy()
    env["SPEDAS_DATA_DIR"] = str(Path(args.spedas_dir))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    failed = []
    for index, month in enumerate(ranges, start=1):
        marker = status_path(state_dir, month)
        result = expected_result(valid_ranges_dir, month)
        prior = {}
        if marker.is_file():
            try:
                prior = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                prior = {}
        if not args.force and compatible_complete(
            prior, notebook_hash, runner_hash, result
        ):
            print(f"[{index}/{len(ranges)}] {month.key}: skip complete")
            continue

        attempt = int(prior.get("attempt", 0)) + 1
        base = {
            "month": month.key,
            "start_time": month.start_text,
            "end_time": month.end_text,
            "notebook_sha256": notebook_hash,
            "runner_sha256": runner_hash,
            "result_path": str(result),
            "attempt": attempt,
            "started_at": datetime.now().isoformat(),
        }
        atomic_json(marker, {**base, "status": "running"})
        log_path = state_dir / "logs" / f"{month.key}.log"
        failed_notebook = state_dir / "failed_notebooks" / f"{month.key}.ipynb"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_execute-one",
            "--notebook", str(notebook),
            "--month", month.key,
            "--start", month.start_text,
            "--end", month.end_text,
            "--kernel-name", args.kernel_name,
            "--valid-ranges-dir", str(valid_ranges_dir),
            "--failed-notebook", str(failed_notebook),
        ]
        print(f"[{index}/{len(ranges)}] {month.key}: run attempt {attempt}")
        started = time.monotonic()
        code = run_child(command, log_path, int(args.timeout), env)
        elapsed = time.monotonic() - started
        if code in {0, NO_DATA_EXIT_CODE} and result.is_file():
            state = "no_data" if code == NO_DATA_EXIT_CODE else "complete"
            terminal = {
                **base,
                "status": state,
                "returncode": code,
                "elapsed_sec": elapsed,
                "finished_at": datetime.now().isoformat(),
            }
            print(f"[{index}/{len(ranges)}] {month.key}: {state} ({elapsed / 60:.1f} min)")
        else:
            state = "timeout" if code == 124 else "failed"
            terminal = {
                **base,
                "status": state,
                "returncode": code,
                "elapsed_sec": elapsed,
                "finished_at": datetime.now().isoformat(),
                "log_path": str(log_path),
                "failed_notebook": str(failed_notebook),
            }
            failed.append(month.key)
            print(f"[{index}/{len(ranges)}] {month.key}: {state} ({elapsed / 60:.1f} min)")
        atomic_json(marker, terminal)

    if failed:
        print(f"failed months: {failed}")
        return 1
    print("all selected months complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Execute the notebook month by month")
    run.add_argument("--start", default=DEFAULT_START)
    run.add_argument("--end", default=DEFAULT_END)
    run.add_argument("--month", action="append", help="Run only YYYYMM; repeatable")
    run.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    run.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    run.add_argument("--valid-ranges-dir", type=Path, default=DEFAULT_VALID_RANGES_DIR)
    run.add_argument("--spedas-dir", type=Path, default=DEFAULT_SPEDAS_DIR)
    run.add_argument("--kernel-name", default="python3")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    one = subparsers.add_parser("_execute-one")
    one.add_argument("--notebook", required=True)
    one.add_argument("--month", required=True)
    one.add_argument("--start", required=True)
    one.add_argument("--end", required=True)
    one.add_argument("--kernel-name", default="python3")
    one.add_argument("--valid-ranges-dir", required=True)
    one.add_argument("--failed-notebook", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "_execute-one":
            return execute_one(args)
        return run_months(args)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
