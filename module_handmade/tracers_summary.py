"""Daily-cache based fixed-layout summary plots for TRACERS."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import multiprocessing as mp
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import time
import textwrap
from typing import Any, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import xarray as xr

from cdasws import CdasWs

from .tracers_loader import (
    KNOWN_DATASETS,
    TracersNoDataError,
    TracersSourceVersion,
    get_tracers_inventory,
    get_tracers_source_version,
    load_tracers_orbit,
    load_tracers_xarray,
    save_tracers_dataset,
    tracers_daily_path,
    tracers_orbit_daily_path,
    tracers_summary_plot_path,
    tracers_summary_trange,
)
from .tracers_conjugacy import (
    CONJUGATE_SATELLITE_REGISTRY,
    compute_conjugate_satellite_day,
    compute_tracers_sm_track,
    conjugacy_daily_path,
    normalize_conjugate_satellites,
)


__all__ = [
    "DailySummaryData",
    "load_daily_summary_data",
    "make_one_hour_summary_plot",
    "make_two_hour_summary_plot",
    "generate_daily_summary_plots",
    "generate_summary_plots_for_date_range",
]


@dataclass
class DailySummaryData:
    probe: int
    date: str
    bin_interval_seconds: float | None
    products: dict[str, xr.Dataset]
    product_errors: dict[str, str]
    orbit: xr.Dataset | None
    orbit_error: str | None
    cache_paths: dict[str, Path]
    conjugate_products: dict[str, xr.Dataset] = field(default_factory=dict)
    conjugate_errors: dict[str, str] = field(default_factory=dict)
    conjugate_satellites: tuple[str, ...] = ()


def _normalize_date(date: str | pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(date)
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    return day.normalize()


def _daily_trange(day: pd.Timestamp) -> list[str]:
    return [
        day.isoformat().replace("+00:00", "Z"),
        (day + pd.Timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    ]


def _resolve_or_none(instrument: str, probe: int, datatype: str) -> str | None:
    key = (
        instrument.strip().lower(),
        int(probe),
        datatype.strip().lower().replace("_", "-"),
    )
    return KNOWN_DATASETS.get(key)


def _product_specs(probe: int) -> dict[str, dict[str, Any]]:
    prefix = f"ts{probe}_l2"
    return {
        "ace": {
            "dataset": _resolve_or_none("ace", probe, "def"),
            "groups": [[f"{prefix}_ace_def"]],
        },
        "aci": {
            "dataset": _resolve_or_none("aci", probe, "ipd"),
            "groups": [[f"{prefix}_aci_tscs_def"]],
        },
        "eac": {
            "dataset": _resolve_or_none("efi", probe, "eac"),
            "groups": [[f"{prefix}_eac_x_spec"], [f"{prefix}_eac_y_spec"]],
        },
        "ehf": {
            "dataset": _resolve_or_none("efi", probe, "ehf"),
            "groups": [[f"{prefix}_hf_spec"]],
        },
        "vdc": {
            "dataset": _resolve_or_none("efi", probe, "vdc"),
            "groups": [[
                f"{prefix}_vdc_xminus",
                f"{prefix}_vdc_xplus",
                f"{prefix}_vdc_yminus",
                f"{prefix}_vdc_yplus",
            ]],
        },
        "msc": {
            "dataset": _resolve_or_none("msc", probe, "bac"),
            "groups": [[f"{prefix}_bac_tscs"]],
        },
        "magic": {
            "dataset": _resolve_or_none("magic", probe, "bdc-16sps"),
            "groups": [[f"{prefix}_magic_gei2000_bdc"]],
        },
    }


def _regular_bin_centers(
    day: pd.Timestamp, n_time: int, bin_interval_seconds: float
) -> np.ndarray:
    start = day + pd.to_timedelta(bin_interval_seconds / 2.0, unit="s")
    return pd.date_range(
        start=start,
        periods=n_time,
        freq=pd.to_timedelta(bin_interval_seconds, unit="s"),
    ).tz_localize(None).to_numpy(dtype="datetime64[ns]")


def _compact_part(
    ds: xr.Dataset,
    variables: list[str],
    day: pd.Timestamp,
    bin_interval_seconds: float,
) -> xr.Dataset:
    compact = ds[variables].copy(deep=False)
    compact.attrs = dict(ds.attrs)
    for name in variables:
        da = compact[name]
        first_dim = da.dims[0]
        has_datetime = (
            first_dim in da.coords
            and np.issubdtype(da.coords[first_dim].dtype, np.datetime64)
        )
        if not has_datetime:
            compact = compact.assign_coords(
                {
                    first_dim: _regular_bin_centers(
                        day, da.sizes[first_dim], bin_interval_seconds
                    )
                }
            )
    return compact


def _download_summary_product(
    key: str,
    spec: dict[str, Any],
    day: pd.Timestamp,
    day_trange: list[str],
    interval: float | None,
    request_timeout: float,
) -> tuple[str, xr.Dataset | None, str | None, str]:
    """Download and compact one product with finite network timeouts."""
    dataset_id = spec["dataset"]
    assert dataset_id is not None
    try:
        try:
            inventory = get_tracers_inventory(
                dataset_id,
                client=CdasWs(
                    timeout=float(request_timeout), disable_cache=True
                ),
                request_timeout=request_timeout,
            )
            overlaps = inventory[
                (inventory["start"] < day + pd.Timedelta(days=1))
                & (inventory["stop"] > day)
            ]
        except Exception:
            overlaps = None
        if overlaps is not None and overlaps.empty:
            return key, None, "No data in selected UTC day.", "inventory"

        bin_data = None
        if interval is not None:
            bin_data = {
                "interval": interval,
                "interpolateMissingValues": False,
                "overrideDefaultBinning": True,
            }
        parts = []
        for group in spec["groups"]:
            loaded = load_tracers_xarray(
                dataset_id,
                day_trange,
                variables=group,
                bin_data=bin_data,
                client=CdasWs(
                    timeout=float(request_timeout), disable_cache=True
                ),
                max_attempts=1,
                retry_wait=0.0,
                request_timeout=request_timeout,
            )
            if interval is None:
                parts.append(loaded[group].copy(deep=False))
            else:
                parts.append(_compact_part(loaded, group, day, interval))
        daily = xr.merge(parts, compat="override", combine_attrs="override")
        daily.attrs.update(parts[0].attrs)
        daily.attrs["tracers_daily_cache_date_utc"] = day.strftime("%Y-%m-%d")
        daily.attrs["tracers_daily_cache_cadence"] = (
            "native" if interval is None else f"{interval:g}s-bin"
        )
        if interval is not None:
            daily.attrs["tracers_daily_cache_bin_seconds"] = interval
        source_version = spec.get("source_version")
        if source_version is not None:
            daily.attrs["tracers_source_version"] = source_version.version
            daily.attrs["tracers_source_files"] = json.dumps(
                list(source_version.files)
            )
            daily.attrs["tracers_source_last_modified"] = json.dumps(
                list(source_version.last_modified)
            )
            daily.attrs["tracers_version_resolution"] = (
                "CDAWeb get_original_files"
            )
        return key, daily, None, "CDAWeb"
    except Exception as exc:
        message = f"Download failed: {type(exc).__name__}: {exc}"
        return key, None, message, "failed"


def _download_product_process(
    send_connection: Any,
    key: str,
    spec: dict[str, Any],
    day: pd.Timestamp,
    day_trange: list[str],
    interval: float | None,
    request_timeout: float,
    cache_path_text: str,
) -> None:
    """Subprocess target: download one product and atomically save its cache."""
    try:
        result_key, daily, error, source = _download_summary_product(
            key,
            spec,
            day,
            day_trange,
            interval,
            request_timeout,
        )
        if daily is not None and error is None:
            cache_path = Path(cache_path_text)
            save_tracers_dataset(daily, cache_path, overwrite=True)
            message = {
                "key": result_key,
                "status": "ok",
                "source": source,
                "error": None,
            }
        elif source == "inventory":
            message = {
                "key": result_key,
                "status": "no_data",
                "source": source,
                "error": error,
            }
        else:
            message = {
                "key": result_key,
                "status": "error",
                "source": source,
                "error": error or "Download returned no dataset.",
            }
    except BaseException as exc:
        message = {
            "key": key,
            "status": "error",
            "source": "subprocess",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        send_connection.send(message)
    finally:
        send_connection.close()


def _remove_partial_cache(cache_path: Path) -> None:
    partial = cache_path.with_name(f".{cache_path.name}.part")
    partial.unlink(missing_ok=True)


def _source_version_process(
    send_connection: Any,
    key: str,
    dataset_id: str,
    day_trange: list[str],
    request_timeout: float,
) -> None:
    """Subprocess target for one bounded CDAWeb source-version query."""
    try:
        resolved = get_tracers_source_version(
            dataset_id,
            day_trange,
            request_timeout=request_timeout,
            max_attempts=1,
            retry_wait=0.0,
        )
        message = {
            "key": key,
            "status": "ok",
            "dataset_id": resolved.dataset_id,
            "version": resolved.version,
            "files": resolved.files,
            "last_modified": resolved.last_modified,
            "error": None,
        }
    except TracersNoDataError as exc:
        message = {
            "key": key,
            "status": "no_data",
            "error": str(exc),
        }
    except BaseException as exc:
        message = {
            "key": key,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        send_connection.send(message)
    finally:
        send_connection.close()


def _resolve_source_versions_with_hard_timeouts(
    candidates: dict[str, dict[str, Any]],
    day_trange: list[str],
    *,
    workers: int,
    request_timeout: float,
    attempt_timeout: float,
    max_attempts: int,
    retry_wait: float,
    verbose: bool,
) -> tuple[
    dict[str, TracersSourceVersion],
    dict[str, str],
    dict[str, str],
]:
    """Resolve current upstream versions without allowing a stuck request."""
    context = mp.get_context("spawn")
    waiting = [
        {
            "key": key,
            "dataset_id": str(spec["dataset"]),
            "attempt": 1,
            "ready_at": time.monotonic(),
        }
        for key, spec in candidates.items()
    ]
    active: dict[str, dict[str, Any]] = {}
    resolved: dict[str, TracersSourceVersion] = {}
    no_data: dict[str, str] = {}
    failures: dict[str, str] = {}

    def schedule_retry(task: dict[str, Any], error: str) -> None:
        attempt = int(task["attempt"])
        key = str(task["key"])
        if attempt >= max_attempts:
            failures[key] = (
                f"Source version check failed after {attempt} attempt(s): "
                f"{error}"
            )
            if verbose:
                print(f"{key:6s}: FAILED {failures[key]}")
            return
        delay = retry_wait * 2 ** (attempt - 1)
        retry_task = dict(task)
        retry_task["attempt"] = attempt + 1
        retry_task["ready_at"] = time.monotonic() + delay
        waiting.append(retry_task)
        if verbose:
            print(
                f"{key:6s}: version attempt {attempt}/{max_attempts} "
                f"failed; retry in {delay:g} s: {error}"
            )

    while waiting or active:
        now = time.monotonic()
        waiting.sort(key=lambda item: float(item["ready_at"]))
        while (
            len(active) < workers
            and waiting
            and float(waiting[0]["ready_at"]) <= now
        ):
            task = waiting.pop(0)
            key = str(task["key"])
            receive_connection, send_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_source_version_process,
                args=(
                    send_connection,
                    key,
                    task["dataset_id"],
                    day_trange,
                    request_timeout,
                ),
                name=f"tracers-version-{key}-a{task['attempt']}",
            )
            process.start()
            send_connection.close()
            task.update(
                {
                    "process": process,
                    "connection": receive_connection,
                    "started_at": time.monotonic(),
                }
            )
            active[key] = task
            if verbose:
                print(
                    f"{key:6s}: version attempt "
                    f"{task['attempt']}/{max_attempts} started"
                )

        for key, task in list(active.items()):
            process = task["process"]
            connection = task["connection"]
            message: dict[str, Any] | None = None
            if connection.poll():
                try:
                    message = connection.recv()
                except EOFError:
                    message = None
            if message is not None:
                connection.close()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                active.pop(key)
                status = str(message["status"])
                if status == "ok":
                    resolved[key] = TracersSourceVersion(
                        dataset_id=str(message["dataset_id"]),
                        version=str(message["version"]),
                        files=tuple(message["files"]),
                        last_modified=tuple(message["last_modified"]),
                    )
                    if verbose:
                        print(f"{key:6s}: source {resolved[key].version}")
                elif status == "no_data":
                    no_data[key] = "No data in selected UTC day."
                    if verbose:
                        print(f"{key:6s}: {no_data[key]}")
                else:
                    schedule_retry(task, str(message.get("error")))
                continue

            elapsed = time.monotonic() - float(task["started_at"])
            if elapsed >= attempt_timeout:
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
                connection.close()
                active.pop(key)
                schedule_retry(
                    task,
                    f"hard timeout after {attempt_timeout:g} s",
                )
                continue

            if not process.is_alive():
                process.join(timeout=1.0)
                connection.close()
                active.pop(key)
                schedule_retry(
                    task,
                    f"subprocess exited with code {process.exitcode} "
                    "without a result",
                )

        if waiting or active:
            time.sleep(0.1)

    return resolved, no_data, failures


def _download_products_with_hard_timeouts(
    downloads: list[tuple[str, dict[str, Any], Path]],
    day: pd.Timestamp,
    day_trange: list[str],
    interval: float | None,
    *,
    workers: int,
    request_timeout: float,
    attempt_timeout: float,
    max_attempts: int,
    retry_wait: float,
    verbose: bool,
) -> dict[str, tuple[str, str | None]]:
    """Run killable product downloads with bounded retries and backoff."""
    context = mp.get_context("spawn")
    waiting = [
        {
            "key": key,
            "spec": spec,
            "cache_path": cache_path,
            "attempt": 1,
            "ready_at": time.monotonic(),
        }
        for key, spec, cache_path in downloads
    ]
    active: dict[str, dict[str, Any]] = {}
    results: dict[str, tuple[str, str | None]] = {}

    def schedule_retry(task: dict[str, Any], error: str) -> None:
        attempt = int(task["attempt"])
        key = str(task["key"])
        if attempt >= max_attempts:
            results[key] = (
                "failed",
                f"Download failed after {attempt} attempt(s): {error}",
            )
            if verbose:
                print(f"{key:6s}: FAILED {results[key][1]}")
            return
        delay = retry_wait * 2 ** (attempt - 1)
        retry_task = dict(task)
        retry_task["attempt"] = attempt + 1
        retry_task["ready_at"] = time.monotonic() + delay
        waiting.append(retry_task)
        if verbose:
            print(
                f"{key:6s}: attempt {attempt}/{max_attempts} failed; "
                f"retry in {delay:g} s: {error}"
            )

    while waiting or active:
        now = time.monotonic()
        waiting.sort(key=lambda item: float(item["ready_at"]))
        while (
            len(active) < workers
            and waiting
            and float(waiting[0]["ready_at"]) <= now
        ):
            task = waiting.pop(0)
            key = str(task["key"])
            receive_connection, send_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_download_product_process,
                args=(
                    send_connection,
                    key,
                    task["spec"],
                    day,
                    day_trange,
                    interval,
                    request_timeout,
                    str(task["cache_path"]),
                ),
                name=f"tracers-download-{key}-a{task['attempt']}",
            )
            process.start()
            send_connection.close()
            task.update(
                {
                    "process": process,
                    "connection": receive_connection,
                    "started_at": time.monotonic(),
                }
            )
            active[key] = task
            if verbose:
                print(
                    f"{key:6s}: download attempt "
                    f"{task['attempt']}/{max_attempts} started"
                )

        for key, task in list(active.items()):
            process = task["process"]
            connection = task["connection"]
            message: dict[str, Any] | None = None
            if connection.poll():
                try:
                    message = connection.recv()
                except EOFError:
                    message = None
            if message is not None:
                connection.close()
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                active.pop(key)
                status = str(message["status"])
                error = message.get("error")
                if status == "ok":
                    results[key] = ("ok", None)
                elif status == "no_data":
                    results[key] = ("no_data", str(error))
                else:
                    schedule_retry(task, str(error))
                continue

            elapsed = time.monotonic() - float(task["started_at"])
            if elapsed >= attempt_timeout:
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
                connection.close()
                _remove_partial_cache(Path(task["cache_path"]))
                active.pop(key)
                schedule_retry(
                    task,
                    f"hard timeout after {attempt_timeout:g} s",
                )
                continue

            if not process.is_alive():
                process.join(timeout=1.0)
                connection.close()
                _remove_partial_cache(Path(task["cache_path"]))
                active.pop(key)
                schedule_retry(
                    task,
                    f"subprocess exited with code {process.exitcode} "
                    "without a result",
                )

        if waiting or active:
            time.sleep(0.1)

    return results


def _download_orbit_process(
    send_connection: Any,
    day_trange: list[str],
    probe: int,
    cache_path_text: str,
) -> None:
    """Subprocess target for one SSCWeb orbit attempt."""
    try:
        orbit = load_tracers_orbit(
            day_trange,
            probe=probe,
            max_duration="1D",
            max_attempts=1,
            retry_wait=0.0,
        )
        save_tracers_dataset(orbit, cache_path_text, overwrite=True)
        message = {"status": "ok", "error": None}
    except BaseException as exc:
        message = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        send_connection.send(message)
    finally:
        send_connection.close()


def _download_orbit_with_hard_timeout(
    day_trange: list[str],
    probe: int,
    cache_path: Path,
    *,
    attempt_timeout: float,
    max_attempts: int,
    retry_wait: float,
    verbose: bool,
) -> str | None:
    """Download orbit with killable attempts; return final error or None."""
    context = mp.get_context("spawn")
    last_error = "Orbit download returned no result."
    for attempt in range(1, max_attempts + 1):
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_download_orbit_process,
            args=(send_connection, day_trange, probe, str(cache_path)),
            name=f"tracers-orbit-a{attempt}",
        )
        process.start()
        send_connection.close()
        if verbose:
            print(f"orbit : download attempt {attempt}/{max_attempts} started")
        started_at = time.monotonic()
        message: dict[str, Any] | None = None
        while time.monotonic() - started_at < attempt_timeout:
            if receive_connection.poll(0.1):
                try:
                    message = receive_connection.recv()
                except EOFError:
                    message = None
                break
            if not process.is_alive():
                break
        if message is not None:
            receive_connection.close()
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if message["status"] == "ok":
                return None
            last_error = str(message.get("error") or last_error)
        else:
            timed_out = time.monotonic() - started_at >= attempt_timeout
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            receive_connection.close()
            last_error = (
                f"hard timeout after {attempt_timeout:g} s"
                if timed_out
                else f"subprocess exited with code {process.exitcode} "
                "without a result"
            )
        _remove_partial_cache(cache_path)
        if attempt < max_attempts:
            delay = retry_wait * 2 ** (attempt - 1)
            if verbose:
                print(
                    f"orbit : attempt {attempt}/{max_attempts} failed; "
                    f"retry in {delay:g} s: {last_error}"
                )
            time.sleep(delay)
    return f"Orbit download failed after {max_attempts} attempt(s): {last_error}"


def _compute_conjugacy_process(
    send_connection: Any,
    satellite: str,
    date: str,
    probe: int,
    orbit_path_text: str,
    cache_path_text: str,
    trace_workers: int,
    trace_chunk_size: int,
) -> None:
    """Subprocess target for one satellite's daily T04 footprint cache."""
    try:
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ[variable] = "1"
        if os.name == "posix":
            try:
                os.setsid()
            except OSError:
                pass
        with xr.open_dataset(orbit_path_text) as cached:
            orbit = cached.load()
        product = compute_conjugate_satellite_day(
            satellite,
            date,
            orbit,
            probe=probe,
            trace_workers=trace_workers,
            trace_chunk_size=trace_chunk_size,
        )
        save_tracers_dataset(product, cache_path_text, overwrite=True)
        message = {"status": "ok", "error": None}
    except BaseException as exc:
        message = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        send_connection.send(message)
    finally:
        send_connection.close()


def _terminate_process_tree(process: mp.Process, *, force: bool = False) -> None:
    """Terminate a controller and its process-pool workers on POSIX."""
    if not process.is_alive():
        process.join(timeout=1.0)
        return
    sent_to_group = False
    if os.name == "posix" and process.pid is not None:
        try:
            process_group = os.getpgid(process.pid)
            if process_group == process.pid and process_group != os.getpgrp():
                os.killpg(
                    process_group,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
                sent_to_group = True
        except (ProcessLookupError, PermissionError):
            pass
    if not sent_to_group:
        process.kill() if force else process.terminate()
    process.join(timeout=5.0)


def _compute_conjugacy_with_hard_timeout(
    satellite: str,
    date: str,
    probe: int,
    orbit_path: Path,
    cache_path: Path,
    *,
    trace_workers: int,
    trace_chunk_size: int,
    attempt_timeout: float,
    max_attempts: int,
    retry_wait: float,
    verbose: bool,
) -> str | None:
    """Compute a killable daily conjugacy product; return final error or None."""
    context = mp.get_context("spawn")
    last_error = "Conjugacy subprocess returned no result."
    for attempt in range(1, max_attempts + 1):
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_compute_conjugacy_process,
            args=(
                send_connection,
                satellite,
                date,
                probe,
                str(orbit_path),
                str(cache_path),
                trace_workers,
                trace_chunk_size,
            ),
            name=f"tracers-conjugacy-{satellite.lower()}-a{attempt}",
        )
        process.start()
        send_connection.close()
        if verbose:
            print(
                f"{satellite:6s}: T04 attempt "
                f"{attempt}/{max_attempts} started "
                f"({trace_workers} workers, chunk={trace_chunk_size})"
            )
        started_at = time.monotonic()
        message: dict[str, Any] | None = None
        while time.monotonic() - started_at < attempt_timeout:
            if receive_connection.poll(0.1):
                try:
                    message = receive_connection.recv()
                except EOFError:
                    message = None
                break
            if not process.is_alive():
                break
        if message is not None:
            receive_connection.close()
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if message["status"] == "ok":
                return None
            last_error = str(message.get("error") or last_error)
        else:
            timed_out = time.monotonic() - started_at >= attempt_timeout
            if process.is_alive():
                _terminate_process_tree(process)
            if process.is_alive():
                _terminate_process_tree(process, force=True)
            receive_connection.close()
            last_error = (
                f"hard timeout after {attempt_timeout:g} s"
                if timed_out
                else f"subprocess exited with code {process.exitcode} "
                "without a result"
            )
        _remove_partial_cache(cache_path)
        if attempt < max_attempts:
            delay = retry_wait * 2 ** (attempt - 1)
            if verbose:
                print(
                    f"{satellite:6s}: T04 attempt {attempt}/{max_attempts} "
                    f"failed; retry in {delay:g} s: {last_error}"
                )
            time.sleep(delay)
    return (
        f"T04 footprint failed after {max_attempts} attempt(s): {last_error}"
    )


def _version_numbers(value: object) -> tuple[int, ...] | None:
    """Parse a scalar/list-like CDF version and ignore trailing zeros."""
    candidates: list[object]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded = value
        candidates = list(decoded) if isinstance(decoded, list) else [decoded]
    elif isinstance(value, (list, tuple, np.ndarray)):
        candidates = list(value)
    else:
        candidates = [value]
    parsed: set[tuple[int, ...]] = set()
    for candidate in candidates:
        text = str(candidate).strip().strip('"').strip("'")
        if text.lower().startswith("v"):
            text = text[1:]
        if not text or any(not part.isdigit() for part in text.split(".")):
            continue
        numbers = [int(part) for part in text.split(".")]
        while len(numbers) > 1 and numbers[-1] == 0:
            numbers.pop()
        parsed.add(tuple(numbers))
    if len(parsed) != 1:
        return None
    return next(iter(parsed))


def _cache_matches_source_version(ds: xr.Dataset, version: str) -> bool:
    cached = ds.attrs.get("tracers_source_version", ds.attrs.get("Data_version"))
    return _version_numbers(cached) == _version_numbers(version)


def _best_local_daily_cache(
    dataset_id: str,
    day: pd.Timestamp,
    interval: float | None,
) -> Path | None:
    """Return the highest local source version, then the legacy cache."""
    legacy = tracers_daily_path(
        dataset_id,
        day,
        bin_interval_seconds=interval,
        source_version=None,
    )
    bin_tag = "native" if interval is None else f"{interval:g}s".replace(".", "p")
    pattern = (
        f"{dataset_id.lower()}_{day.strftime('%Y%m%d')}_v*_{bin_tag}.nc"
    )
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for path in legacy.parent.glob(pattern):
        match = re.search(r"_v(\d+(?:\.\d+)*)_", path.name)
        if match is None:
            continue
        version = _version_numbers(match.group(1))
        if version is not None:
            candidates.append((version, path))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    return legacy if legacy.exists() else None


def load_daily_summary_data(
    date: str | pd.Timestamp,
    *,
    probe: int = 2,
    bin_interval_seconds: float | None = None,
    conjugate_satellites: str | Sequence[str] | None = None,
    force_download: bool = False,
    save_daily_cache: bool = True,
    download_workers: int = 3,
    http_timeout_seconds: float = 60.0,
    download_attempt_timeout_seconds: float = 300.0,
    conjugacy_attempt_timeout_seconds: float = 1800.0,
    conjugacy_workers: int = 4,
    conjugacy_chunk_size: int = 48,
    download_max_attempts: int = 3,
    download_retry_wait_seconds: float = 10.0,
    verbose: bool = True,
) -> DailySummaryData:
    """Load/create one UTC day of preferably native-cadence products and orbit.

    ``bin_interval_seconds=None`` (default) requests native CDAWeb samples.
    A positive number remains available for explicit overview binning and for
    reading historical binned caches.
    """
    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    interval = (
        None
        if bin_interval_seconds is None
        else float(bin_interval_seconds)
    )
    if interval is not None and interval <= 0:
        raise ValueError("bin_interval_seconds must be positive or None.")
    conjugate_keys = normalize_conjugate_satellites(conjugate_satellites)
    download_workers = int(download_workers)
    if not 1 <= download_workers <= 4:
        raise ValueError("download_workers must be between 1 and 4.")
    http_timeout_seconds = float(http_timeout_seconds)
    download_attempt_timeout_seconds = float(download_attempt_timeout_seconds)
    conjugacy_attempt_timeout_seconds = float(conjugacy_attempt_timeout_seconds)
    conjugacy_workers = int(conjugacy_workers)
    conjugacy_chunk_size = int(conjugacy_chunk_size)
    download_max_attempts = int(download_max_attempts)
    download_retry_wait_seconds = float(download_retry_wait_seconds)
    if http_timeout_seconds <= 0:
        raise ValueError("http_timeout_seconds must be positive.")
    if download_attempt_timeout_seconds <= http_timeout_seconds:
        raise ValueError(
            "download_attempt_timeout_seconds must exceed http_timeout_seconds."
        )
    if conjugacy_attempt_timeout_seconds <= 0:
        raise ValueError("conjugacy_attempt_timeout_seconds must be positive.")
    if not 1 <= conjugacy_workers <= 16:
        raise ValueError("conjugacy_workers must be between 1 and 16.")
    if conjugacy_chunk_size <= 0:
        raise ValueError("conjugacy_chunk_size must be positive.")
    if not 1 <= download_max_attempts <= 5:
        raise ValueError("download_max_attempts must be between 1 and 5.")
    if download_retry_wait_seconds < 0:
        raise ValueError("download_retry_wait_seconds must be non-negative.")

    day = _normalize_date(date)
    day_text = day.strftime("%Y-%m-%d")
    day_trange = _daily_trange(day)
    specs = _product_specs(probe)
    products: dict[str, xr.Dataset] = {}
    errors: dict[str, str] = {}
    paths: dict[str, Path] = {}

    version_candidates = {
        key: spec
        for key, spec in specs.items()
        if spec["dataset"] is not None
    }
    if version_candidates:
        (
            source_versions,
            version_no_data,
            version_check_failures,
        ) = _resolve_source_versions_with_hard_timeouts(
            version_candidates,
            day_trange,
            workers=min(download_workers, len(version_candidates)),
            request_timeout=http_timeout_seconds,
            attempt_timeout=download_attempt_timeout_seconds,
            max_attempts=download_max_attempts,
            retry_wait=download_retry_wait_seconds,
            verbose=verbose,
        )
        errors.update(version_no_data)
    else:
        source_versions = {}
        version_check_failures = {}

    downloads: list[tuple[str, dict[str, Any], Path]] = []
    for key, original_spec in specs.items():
        spec = dict(original_spec)
        dataset_id = spec["dataset"]
        if dataset_id is None:
            errors[key] = f"Dataset not registered for TS{probe}."
            if verbose:
                print(f"{key:6s}: {errors[key]}")
            continue
        if key in version_check_failures:
            fallback_path = _best_local_daily_cache(dataset_id, day, interval)
            if fallback_path is not None and not force_download:
                try:
                    with xr.open_dataset(fallback_path) as cached:
                        fallback = cached.load()
                    fallback.attrs["tracers_version_warning"] = (
                        version_check_failures[key]
                        + "; local cache freshness is unverified"
                    )
                    products[key] = fallback
                    paths[key] = fallback_path
                    if verbose:
                        print(
                            f"{key:6s}: WARNING using local cache because "
                            "current source version could not be verified"
                        )
                    continue
                except Exception as exc:
                    version_check_failures[key] += (
                        f"; local cache read failed: {type(exc).__name__}: {exc}"
                    )
            errors[key] = version_check_failures[key]
            continue
        if key not in source_versions:
            continue
        source_version = source_versions[key]
        spec["source_version"] = source_version
        cache_path = tracers_daily_path(
            dataset_id,
            day,
            bin_interval_seconds=interval,
            source_version=source_version.version,
        )
        paths[key] = cache_path
        if cache_path.exists() and not force_download:
            try:
                with xr.open_dataset(cache_path) as cached:
                    daily = cached.load()
                if _cache_matches_source_version(daily, source_version.version):
                    products[key] = daily
                    if verbose:
                        print(
                            f"{key:6s}: cache {source_version.version} "
                            f"{dict(daily.sizes)}"
                        )
                    continue
                if verbose:
                    print(
                        f"{key:6s}: cache filename/version mismatch; refreshing"
                    )
            except Exception as exc:
                if verbose:
                    print(
                        f"{key:6s}: cache read failed; refreshing: "
                        f"{type(exc).__name__}: {exc}"
                    )

        legacy_path = tracers_daily_path(
            dataset_id,
            day,
            bin_interval_seconds=interval,
            source_version=None,
        )
        if not force_download and legacy_path.exists():
            try:
                with xr.open_dataset(legacy_path) as cached:
                    legacy = cached.load()
                if _cache_matches_source_version(legacy, source_version.version):
                    legacy.attrs["tracers_source_version"] = source_version.version
                    legacy.attrs["tracers_source_files"] = json.dumps(
                        list(source_version.files)
                    )
                    legacy.attrs["tracers_source_last_modified"] = json.dumps(
                        list(source_version.last_modified)
                    )
                    legacy.attrs["tracers_version_resolution"] = (
                        "CDAWeb get_original_files; migrated without redownload"
                    )
                    if save_daily_cache:
                        save_tracers_dataset(legacy, cache_path, overwrite=True)
                    products[key] = legacy
                    if verbose:
                        print(
                            f"{key:6s}: migrated {source_version.version} "
                            "cache without download"
                        )
                    continue
            except Exception as exc:
                if verbose:
                    print(
                        f"{key:6s}: legacy cache validation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        downloads.append((key, spec, cache_path))

    if downloads and save_daily_cache:
        download_results = _download_products_with_hard_timeouts(
            downloads,
            day,
            day_trange,
            interval,
            workers=min(download_workers, len(downloads)),
            request_timeout=http_timeout_seconds,
            attempt_timeout=download_attempt_timeout_seconds,
            max_attempts=download_max_attempts,
            retry_wait=download_retry_wait_seconds,
            verbose=verbose,
        )
        for key, _, cache_path in downloads:
            status, error = download_results[key]
            if status == "ok":
                try:
                    with xr.open_dataset(cache_path) as cached:
                        daily = cached.load()
                    products[key] = daily
                    if verbose:
                        print(f"{key:6s}: CDAWeb {dict(daily.sizes)}")
                except Exception as exc:
                    errors[key] = (
                        f"Cache read failed after download: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                errors[key] = error or "Download returned no dataset."
                if verbose and status == "no_data":
                    print(f"{key:6s}: {errors[key]}")
    elif downloads:
        for key, spec, _ in downloads:
            last_error = "Download returned no dataset."
            for attempt in range(1, download_max_attempts + 1):
                _, daily, error, source = _download_summary_product(
                    key,
                    spec,
                    day,
                    day_trange,
                    interval,
                    http_timeout_seconds,
                )
                if daily is not None and error is None:
                    products[key] = daily
                    if verbose:
                        print(f"{key:6s}: {source:6s} {dict(daily.sizes)}")
                    break
                last_error = error or last_error
                if source == "inventory":
                    break
                if attempt < download_max_attempts:
                    delay = download_retry_wait_seconds * 2 ** (attempt - 1)
                    time.sleep(delay)
            else:
                daily = None
            if key not in products:
                errors[key] = last_error
                if verbose:
                    print(f"{key:6s}: {last_error}")

    orbit_path = tracers_orbit_daily_path(probe, day)
    paths["orbit"] = orbit_path
    orbit: xr.Dataset | None = None
    orbit_error: str | None = None
    cached_orbit: xr.Dataset | None = None
    try:
        if orbit_path.exists() and not force_download:
            with xr.open_dataset(orbit_path) as cached:
                cached_orbit = cached.load()
            if "SignedInvariantLatitude" in cached_orbit:
                orbit = cached_orbit
                source = "cache"
            elif verbose:
                print("orbit : legacy cache lacks signed invariant latitude; refreshing")
        if orbit is None:
            orbit_download_path = (
                orbit_path
                if save_daily_cache
                else Path(tempfile.gettempdir())
                / f"tracers{probe}_orbit_{day.strftime('%Y%m%d')}_{os.getpid()}.nc"
            )
            final_error = _download_orbit_with_hard_timeout(
                day_trange,
                probe,
                orbit_download_path,
                attempt_timeout=download_attempt_timeout_seconds,
                max_attempts=download_max_attempts,
                retry_wait=download_retry_wait_seconds,
                verbose=verbose,
            )
            if final_error is not None:
                raise RuntimeError(final_error)
            with xr.open_dataset(orbit_download_path) as downloaded:
                orbit = downloaded.load()
            if not save_daily_cache:
                orbit_download_path.unlink(missing_ok=True)
            source = "SSCWeb"
        if verbose:
            print(f"orbit : {source:6s} {dict(orbit.sizes)}")
    except Exception as exc:
        orbit_error = f"{type(exc).__name__}: {exc}"
        orbit = cached_orbit
        if verbose:
            fallback = " (using legacy cache)" if orbit is not None else ""
            print(f"orbit : FAILED {orbit_error}{fallback}")

    sm_track_names = {
        "Position_GSM_RE",
        "Position_SM_RE",
        "MLT_SM",
        "MagneticLatitude_SM",
        "Hemisphere_SM",
    }
    if conjugate_keys and orbit is not None and not sm_track_names.issubset(orbit):
        try:
            sm_track = compute_tracers_sm_track(orbit)
            for name in sm_track.data_vars:
                orbit[name] = sm_track[name]
            orbit.attrs["tracers_sm_track"] = (
                "geopack-vectorize GEO->GSM->SM; independent of conjugate satellite"
            )
            if save_daily_cache:
                save_tracers_dataset(orbit, orbit_path, overwrite=True)
            if verbose:
                print("orbit : added independent TRACERS SM MLT/latitude track")
        except Exception as exc:
            orbit.attrs["tracers_sm_track_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            if verbose:
                print(
                    "orbit : WARNING independent SM track failed; "
                    "plot will use SSCWeb MLT/signed invariant latitude: "
                    f"{type(exc).__name__}: {exc}"
                )

    conjugate_products: dict[str, xr.Dataset] = {}
    conjugate_errors: dict[str, str] = {}
    conjugate_downloads: list[tuple[str, Path]] = []
    for satellite in conjugate_keys:
        cache_path = conjugacy_daily_path(satellite, probe, day)
        paths[f"conjugate:{satellite}"] = cache_path
        if cache_path.exists() and not force_download:
            try:
                with xr.open_dataset(cache_path) as cached:
                    conjugate_products[satellite] = cached.load()
                if verbose:
                    print(
                        f"{satellite:6s}: T04 cache "
                        f"{dict(conjugate_products[satellite].sizes)}"
                    )
                continue
            except Exception as exc:
                if verbose:
                    print(
                        f"{satellite:6s}: T04 cache read failed; recomputing: "
                        f"{type(exc).__name__}: {exc}"
                    )
        conjugate_downloads.append((satellite, cache_path))

    if conjugate_downloads and orbit is None:
        for satellite, _ in conjugate_downloads:
            conjugate_errors[satellite] = (
                "T04 footprint unavailable because TRACERS orbit is unavailable."
            )
    elif conjugate_downloads:
        orbit_snapshot_path = (
            Path(tempfile.gettempdir())
            / f"tracers{probe}_orbit_for_conjugacy_"
            f"{day.strftime('%Y%m%d')}_{os.getpid()}.nc"
        )
        save_tracers_dataset(orbit, orbit_snapshot_path, overwrite=True)
        try:
            for satellite, cache_path in conjugate_downloads:
                output_path = (
                    cache_path
                    if save_daily_cache
                    else Path(tempfile.gettempdir())
                    / f"{satellite.lower()}_t04_{day.strftime('%Y%m%d')}_"
                    f"{os.getpid()}.nc"
                )
                final_error = _compute_conjugacy_with_hard_timeout(
                    satellite,
                    day_text,
                    probe,
                    orbit_snapshot_path,
                    output_path,
                    trace_workers=conjugacy_workers,
                    trace_chunk_size=conjugacy_chunk_size,
                    attempt_timeout=conjugacy_attempt_timeout_seconds,
                    max_attempts=download_max_attempts,
                    retry_wait=download_retry_wait_seconds,
                    verbose=verbose,
                )
                if final_error is not None:
                    conjugate_errors[satellite] = final_error
                    if verbose:
                        print(f"{satellite:6s}: FAILED {final_error}")
                    continue
                try:
                    with xr.open_dataset(output_path) as computed:
                        conjugate_products[satellite] = computed.load()
                    if verbose:
                        print(
                            f"{satellite:6s}: T04      "
                            f"{dict(conjugate_products[satellite].sizes)}"
                        )
                except Exception as exc:
                    conjugate_errors[satellite] = (
                        f"T04 cache read failed after computation: "
                        f"{type(exc).__name__}: {exc}"
                    )
                finally:
                    if not save_daily_cache:
                        output_path.unlink(missing_ok=True)
        finally:
            orbit_snapshot_path.unlink(missing_ok=True)

    return DailySummaryData(
        probe=probe,
        date=day_text,
        bin_interval_seconds=interval,
        products=products,
        product_errors=errors,
        orbit=orbit,
        orbit_error=orbit_error,
        cache_paths=paths,
        conjugate_products=conjugate_products,
        conjugate_errors=conjugate_errors,
        conjugate_satellites=conjugate_keys,
    )


def _trim_datetime_dims(ds: xr.Dataset, trange: tuple[str, str]) -> xr.Dataset:
    bounds = tuple(
        pd.Timestamp(value).tz_convert("UTC").tz_localize(None).to_datetime64()
        for value in trange
    )
    trimmed = ds
    for dim in list(trimmed.dims):
        if (
            dim in trimmed.coords
            and np.issubdtype(trimmed[dim].dtype, np.datetime64)
        ):
            trimmed = trimmed.sel({dim: slice(bounds[0], bounds[1])})
    return trimmed


def _time_values(da: xr.DataArray) -> np.ndarray:
    first_dim = da.dims[0]
    if (
        first_dim in da.coords
        and np.issubdtype(da[first_dim].dtype, np.datetime64)
    ):
        return np.asarray(da[first_dim].values)
    raise ValueError(f"No datetime coordinate on {da.name!r}.")


def _positive_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("No finite positive values in selected interval.")
    vmin, vmax = np.nanpercentile(finite, [5.0, 99.5])
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(vmin) * 10.0
    return float(vmin), float(vmax)


def _cadence_seconds(
    times: np.ndarray, nominal_seconds: float | None = None
) -> float:
    """Return a robust native cadence used only to detect/display gaps."""
    if nominal_seconds is not None:
        cadence = float(nominal_seconds)
        if cadence <= 0:
            raise ValueError("nominal cadence must be positive.")
        return cadence
    values = np.asarray(times, dtype="datetime64[ns]")
    if values.size < 2:
        return 1.0
    differences = np.diff(values).astype("timedelta64[ns]").astype(float) / 1e9
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        return 1.0
    return float(np.nanmedian(differences))


def _contiguous_slices(
    times: np.ndarray, nominal_seconds: float | None = None
) -> tuple[list[slice], float]:
    """Split native samples without interpolating over missing acquisitions."""
    values = np.asarray(times, dtype="datetime64[ns]")
    cadence = _cadence_seconds(values, nominal_seconds)
    if values.size == 0:
        return [], cadence
    differences = np.diff(values).astype("timedelta64[ns]").astype(float) / 1e9
    breaks = np.flatnonzero(
        ~np.isfinite(differences) | (differences > 1.5 * cadence)
    ) + 1
    bounds = np.concatenate(([0], breaks, [values.size]))
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])], cadence


def _time_edges(times: np.ndarray, cadence_seconds: float) -> np.ndarray:
    """Cell edges for one already gap-free native-time segment."""
    values = np.asarray(times, dtype="datetime64[ns]")
    half = np.timedelta64(max(1, int(round(cadence_seconds * 5e8))), "ns")
    if values.size == 1:
        return np.asarray([values[0] - half, values[0] + half])
    middle_ns = (
        values[:-1].astype("datetime64[ns]").astype(np.int64)
        + values[1:].astype("datetime64[ns]").astype(np.int64)
    ) // 2
    middle = middle_ns.astype("datetime64[ns]")
    return np.concatenate(([values[0] - half], middle, [values[-1] + half]))


def _positive_axis_edges(values: np.ndarray) -> np.ndarray:
    """Geometric cell edges for a positive, increasing log-scale axis."""
    y = np.asarray(values, dtype=float)
    if y.size == 1:
        factor = np.sqrt(2.0)
        return np.asarray([y[0] / factor, y[0] * factor])
    middle = np.sqrt(y[:-1] * y[1:])
    first = y[0] ** 2 / middle[0]
    last = y[-1] ** 2 / middle[-1]
    return np.concatenate(([first], middle, [last]))


def _plot_spectrogram(
    ax: plt.Axes,
    da: xr.DataArray,
    title: str,
    ylabel: str,
    *,
    nominal_cadence_seconds: float | None = None,
):
    time_dim, spectral_dim = da.dims[:2]
    values = np.asarray(da.transpose(time_dim, spectral_dim).values, dtype=float)
    y = np.asarray(da[spectral_dim].values, dtype=float)
    keep = np.isfinite(y) & (y > 0)
    values = np.where(np.isfinite(values) & (values > 0), values, np.nan)[
        :, keep
    ]
    y = y[keep]
    order = np.argsort(y)
    y = y[order]
    values = values[:, order]
    vmin, vmax = _positive_limits(values)
    times = _time_values(da)
    segments, cadence = _contiguous_slices(times, nominal_cadence_seconds)
    mesh = None
    y_edges = _positive_axis_edges(y)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for segment in segments:
        candidate = ax.pcolormesh(
            _time_edges(times[segment], cadence),
            y_edges,
            values[segment].T,
            shading="flat",
            norm=norm,
            cmap="turbo",
        )
        if mesh is None:
            mesh = candidate
    if mesh is None:
        raise ValueError("No time samples in selected interval.")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    return mesh


def _plot_gap_aware_line(
    ax: plt.Axes,
    da: xr.DataArray,
    *,
    nominal_cadence_seconds: float | None = None,
    label: str | None = None,
    **plot_kwargs: Any,
) -> None:
    """Plot native samples while leaving acquisition gaps visibly blank."""
    times = _time_values(da)
    segments, _ = _contiguous_slices(times, nominal_cadence_seconds)
    for index, segment in enumerate(segments):
        ax.plot(
            times[segment],
            np.asarray(da.values)[segment],
            label=label if index == 0 else None,
            marker="." if segment.stop - segment.start == 1 else None,
            markersize=2,
            **plot_kwargs,
        )


def _unavailable(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title, loc="left", fontsize=10)
    ax.text(
        0.5,
        0.5,
        textwrap.fill(str(message), 100),
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_yticks([])


def _availability_note(ax: plt.Axes, message: str) -> None:
    """Annotate a missing comparison series without blanking valid data."""
    ax.text(
        0.99,
        0.05,
        textwrap.fill(str(message), 80),
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.8},
    )


def _mlt_without_wraps(values: xr.DataArray) -> np.ndarray:
    mlt = np.asarray(values.values, dtype=float).copy()
    wraps = np.abs(np.diff(mlt)) >= 12.0
    mlt[1:][wraps] = np.nan
    return mlt


def _conjugacy_values_for_plot(
    values: xr.DataArray,
    hemisphere: xr.DataArray,
    *,
    cyclic_mlt: bool = False,
) -> xr.DataArray:
    """Break MLT wraps and north/south branch switches without interpolation."""
    plotted = values.astype(float).copy()
    array = np.asarray(plotted.values, dtype=float).copy()
    hemi = np.asarray(hemisphere.values, dtype=float)
    if array.size > 1:
        switches = (
            np.isfinite(hemi[:-1])
            & np.isfinite(hemi[1:])
            & (hemi[:-1] != hemi[1:])
        )
        array[1:][switches] = np.nan
        if cyclic_mlt:
            wraps = np.abs(np.diff(array)) >= 12.0
            array[1:][wraps] = np.nan
    plotted.values = array
    return plotted


def make_one_hour_summary_plot(
    daily: DailySummaryData,
    start_hour_utc: int,
    *,
    save: bool = True,
    close: bool = True,
) -> tuple[plt.Figure, Path]:
    """Trim one daily cache to a one-hour slot and make the fixed 10-row plot."""
    trange = tracers_summary_trange(daily.date, start_hour_utc)
    products = {
        key: _trim_datetime_dims(ds, trange)
        for key, ds in daily.products.items()
    }
    orbit = (
        None
        if daily.orbit is None
        else _trim_datetime_dims(daily.orbit, trange)
    )
    conjugate_products = {
        key: _trim_datetime_dims(ds, trange)
        for key, ds in daily.conjugate_products.items()
    }
    probe = daily.probe
    prefix = f"ts{probe}_l2"
    specs = _product_specs(probe)
    fig, axes = plt.subplots(
        10, 1, figsize=(15, 25), sharex=True, constrained_layout=True
    )
    start_plot = pd.Timestamp(trange[0]).to_pydatetime()
    stop_plot = pd.Timestamp(trange[1]).to_pydatetime()

    try:
        da = products["ace"][f"{prefix}_ace_def"]
        da = da.mean(dim=da.dims[-1], skipna=True)
        mesh = _plot_spectrogram(
            axes[0],
            da,
            "(a) ACE electron differential energy flux (look-direction mean)",
            "Energy [eV]",
            nominal_cadence_seconds=6.4,
        )
        fig.colorbar(
            mesh,
            ax=axes[0],
            label="eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$",
        )
    except Exception as exc:
        _unavailable(
            axes[0], "(a) ACE", daily.product_errors.get("ace", str(exc))
        )

    try:
        da = products["aci"][f"{prefix}_aci_tscs_def"]
        da = da.mean(dim=da.dims[-1], skipna=True)
        mesh = _plot_spectrogram(
            axes[1],
            da,
            "(b) ACI ion differential energy flux (look-direction mean)",
            "Energy [eV]",
            nominal_cadence_seconds=40.0,
        )
        fig.colorbar(
            mesh,
            ax=axes[1],
            label="eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$",
        )
    except Exception as exc:
        _unavailable(
            axes[1], "(b) ACI", daily.product_errors.get("aci", str(exc))
        )

    try:
        x_spec = products["eac"][f"{prefix}_eac_x_spec"]
        y_spec = products["eac"][f"{prefix}_eac_y_spec"]
        mesh = _plot_spectrogram(
            axes[2],
            0.5 * (x_spec + y_spec),
            "(c) EFI EAC mean X/Y electric-field PSD",
            "Frequency [Hz]",
            nominal_cadence_seconds=64.0,
        )
        fig.colorbar(
            mesh, ax=axes[2], label="(V m$^{-1}$)$^2$ Hz$^{-1}$"
        )
    except Exception as exc:
        _unavailable(
            axes[2], "(c) EFI EAC", daily.product_errors.get("eac", str(exc))
        )

    try:
        da = products["ehf"][f"{prefix}_hf_spec"]
        mesh = _plot_spectrogram(
            axes[3],
            da,
            "(d) EFI EHF electric-field PSD",
            "Frequency [Hz]",
            # EHF is snapshot data. Give an isolated snapshot a narrow visual
            # footprint; do not stretch it to the next snapshot.
            nominal_cadence_seconds=30.0,
        )
        fig.colorbar(
            mesh, ax=axes[3], label="(V m$^{-1}$)$^2$ Hz$^{-1}$"
        )
    except Exception as exc:
        _unavailable(
            axes[3], "(d) EFI EHF", daily.product_errors.get("ehf", str(exc))
        )

    try:
        ds = products["vdc"]
        for name, color in zip(
            specs["vdc"]["groups"][0],
            ["tab:blue", "tab:cyan", "tab:red", "tab:orange"],
        ):
            da = ds[name]
            _plot_gap_aware_line(
                axes[4],
                da,
                lw=0.9,
                color=color,
                label=name.rsplit("_", 1)[-1],
            )
        axes[4].set_ylabel("Probe voltage [V]")
        axes[4].set_title("(e) EFI VDC probe voltages", loc="left", fontsize=10)
        axes[4].legend(ncol=4, fontsize=8, loc="upper right")
    except Exception as exc:
        _unavailable(
            axes[4], "(e) EFI VDC", daily.product_errors.get("vdc", str(exc))
        )

    try:
        da = products["msc"][f"{prefix}_bac_tscs"]
        rms = np.sqrt((da.astype(float) ** 2).mean(dim=da.dims[1], skipna=True))
        for index, label in enumerate(["Bx", "By", "Bz"]):
            _plot_gap_aware_line(
                axes[5],
                rms.isel({rms.dims[-1]: index}),
                nominal_cadence_seconds=64.0,
                lw=0.9,
                label=label,
            )
        axes[5].set_ylabel("RMS [nT]")
        axes[5].set_title(
            "(f) MSC TSCS native-packet waveform RMS (overview only)",
            loc="left",
            fontsize=10,
        )
        axes[5].legend(ncol=3, fontsize=8, loc="upper right")
    except Exception as exc:
        _unavailable(
            axes[5], "(f) MSC", daily.product_errors.get("msc", str(exc))
        )

    try:
        da = products["magic"][f"{prefix}_magic_gei2000_bdc"]
        for index, label in enumerate(["Bx", "By", "Bz"]):
            _plot_gap_aware_line(
                axes[6],
                da.isel({da.dims[-1]: index}),
                nominal_cadence_seconds=1.0 / 16.0,
                lw=0.9,
                label=label,
            )
        axes[6].set_ylabel("B [nT]")
        axes[6].set_title(
            "(g) MAGIC DC magnetic field (GEI2000; native cadence)",
            loc="left",
            fontsize=10,
        )
        axes[6].legend(ncol=3, fontsize=8, loc="upper right")
    except Exception as exc:
        _unavailable(
            axes[6], "(g) MAGIC", daily.product_errors.get("magic", str(exc))
        )

    if orbit is None:
        _unavailable(
            axes[7], "(h) Altitude", daily.orbit_error or "Orbit unavailable."
        )
    else:
        axes[7].plot(orbit.time, orbit.Altitude, color="black", lw=1.0)
        axes[7].set_ylabel("Altitude [km]")
        axes[7].set_title(
            "(h) Spherical geocentric altitude", loc="left", fontsize=10
        )

    if daily.conjugate_satellites:
        available_keys = [
            key for key in daily.conjugate_satellites if key in conjugate_products
        ]
        reference = (
            conjugate_products[available_keys[0]] if available_keys else None
        )
        tracer_track: tuple[
            xr.DataArray, xr.DataArray, xr.DataArray, str
        ] | None = None
        if (
            orbit is not None
            and {"MLT_SM", "MagneticLatitude_SM", "Hemisphere_SM"}.issubset(
                orbit
            )
        ):
            tracer_track = (
                orbit.MLT_SM,
                orbit.MagneticLatitude_SM,
                orbit.Hemisphere_SM,
                "sm",
            )
        elif reference is not None:
            tracer_track = (
                reference.tracers_mlt,
                reference.tracers_magnetic_latitude,
                reference.selected_hemisphere,
                "sm",
            )
        elif (
            orbit is not None
            and "MLT" in orbit
            and "SignedInvariantLatitude" in orbit
        ):
            fallback_hemisphere = xr.where(
                orbit.SignedInvariantLatitude < 0.0, -1, 1
            )
            tracer_track = (
                orbit.MLT,
                orbit.SignedInvariantLatitude,
                fallback_hemisphere,
                "sscweb_fallback",
            )

        if tracer_track is None:
            message = daily.orbit_error or "TRACERS orbit unavailable."
            _unavailable(axes[8], "(i) TRACERS MLT", message)
            _unavailable(axes[9], "(j) TRACERS magnetic latitude", message)
        else:
            tracer_mlt_raw, tracer_lat_raw, tracer_hemisphere, track_source = (
                tracer_track
            )
            tracer_mlt = _conjugacy_values_for_plot(
                tracer_mlt_raw,
                tracer_hemisphere,
                cyclic_mlt=True,
            )
            tracer_lat = _conjugacy_values_for_plot(
                tracer_lat_raw,
                tracer_hemisphere,
            )
            _plot_gap_aware_line(
                axes[8],
                tracer_mlt,
                nominal_cadence_seconds=60.0,
                color="tab:purple",
                lw=1.0,
                label=f"TRACERS-{probe}",
            )
            _plot_gap_aware_line(
                axes[9],
                tracer_lat,
                nominal_cadence_seconds=60.0,
                color="tab:brown",
                lw=1.0,
                label=f"TRACERS-{probe}",
            )
            for key in available_keys:
                footprint = conjugate_products[key]
                spec = CONJUGATE_SATELLITE_REGISTRY[key]
                footprint_mlt = _conjugacy_values_for_plot(
                    footprint.conjugate_footprint_mlt,
                    footprint.selected_hemisphere,
                    cyclic_mlt=True,
                )
                footprint_lat = _conjugacy_values_for_plot(
                    footprint.conjugate_footprint_magnetic_latitude,
                    footprint.selected_hemisphere,
                )
                _plot_gap_aware_line(
                    axes[8],
                    footprint_mlt,
                    nominal_cadence_seconds=60.0,
                    color=spec.color,
                    ls="--",
                    lw=1.0,
                    label=spec.label,
                )
                _plot_gap_aware_line(
                    axes[9],
                    footprint_lat,
                    nominal_cadence_seconds=60.0,
                    color=spec.color,
                    ls="--",
                    lw=1.0,
                    label=spec.label,
                )
            axes[8].set_ylim(0, 24)
            axes[8].set_yticks([0, 6, 12, 18, 24])
            axes[8].set_ylabel("MLT [h]")
            if track_source == "sm":
                title_i = (
                    "(i) T04 comparison at instantaneous TRACERS altitude: "
                    "MLT (SM)"
                )
                title_j = (
                    "(j) T04 comparison at instantaneous TRACERS altitude: "
                    "magnetic latitude (SM)"
                )
            else:
                title_i = "(i) TRACERS SSCWeb LT_GM; T04 footprint if available"
                title_j = (
                    "(j) TRACERS signed SSCWeb DipInv; T04 footprint if available"
                )
            axes[8].set_title(title_i, loc="left", fontsize=10)
            axes[9].set_ylim(-90, 90)
            axes[9].set_ylabel("Signed magnetic lat. [deg]")
            axes[9].set_title(title_j, loc="left", fontsize=10)
            axes[8].legend(ncol=2, fontsize=8, loc="upper right")
            axes[9].legend(ncol=2, fontsize=8, loc="upper right")
            unavailable_keys = [
                key
                for key in daily.conjugate_satellites
                if key not in available_keys
            ]
            if unavailable_keys:
                message = "; ".join(
                    f"{key}: {daily.conjugate_errors.get(key, 'unavailable')}"
                    for key in unavailable_keys
                )
                _availability_note(axes[8], message)
                _availability_note(axes[9], message)
    elif orbit is None:
        _unavailable(axes[8], "(i) MLT", daily.orbit_error or "Orbit unavailable.")
        _unavailable(
            axes[9],
            "(j) Invariant Latitude",
            daily.orbit_error or "Orbit unavailable.",
        )
    else:
        axes[8].plot(
            orbit.time,
            _mlt_without_wraps(orbit.MLT),
            color="tab:purple",
            lw=1.0,
        )
        axes[8].set_ylim(0, 24)
        axes[8].set_yticks([0, 6, 12, 18, 24])
        axes[8].set_ylabel("MLT [h]")
        axes[8].set_title(
            "(i) SSCWeb LT_GM (not AACGM-recomputed)", loc="left", fontsize=10
        )

        if "SignedInvariantLatitude" in orbit:
            axes[9].plot(
                orbit.time,
                orbit.SignedInvariantLatitude,
                color="tab:brown",
                lw=1.0,
            )
            axes[9].set_ylim(-90, 90)
            axes[9].set_ylabel("Signed inv. lat. [deg]")
            axes[9].set_title(
                "(j) Signed SSCWeb DipInv: hemisphere from Lat_GM",
                loc="left",
                fontsize=10,
            )
        else:
            _unavailable(
                axes[9],
                "(j) Signed Invariant Latitude",
                daily.orbit_error or "Legacy orbit cache lacks SSCWeb Lat_GM.",
            )

    for ax in axes:
        ax.set_xlim(start_plot, stop_plot)
        ax.grid(True, which="major", alpha=0.2)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("UTC")
    instrument_cadence = (
        "native-cadence instrument cache"
        if daily.bin_interval_seconds is None
        else f"{daily.bin_interval_seconds:g} s instrument cache"
    )
    orbit_mode = (
        "T04+IGRF conjugacy at instantaneous TRACERS altitude"
        if daily.conjugate_satellites
        else "native 60 s SSCWeb orbit"
    )
    fig.suptitle(
        f"TRACERS-{probe} L2 summary: {trange[0]} – {trange[1]}\n"
        f"{instrument_cadence}; {orbit_mode}; "
        "no additional quality mask or response correction",
        fontsize=14,
    )
    figure_path = tracers_summary_plot_path(
        daily.date, start_hour_utc, probe=probe
    )
    if save:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    if close:
        plt.close(fig)
    return fig, figure_path


# Backward-compatible name. Slots now follow the one-hour summary specification.
make_two_hour_summary_plot = make_one_hour_summary_plot


def _plot_cached_hour(
    date: str,
    probe: int,
    bin_interval_seconds: float | None,
    start_hour: int,
    cache_paths: dict[str, str],
    product_errors: dict[str, str],
    orbit_error: str | None,
    conjugate_satellites: tuple[str, ...],
    conjugate_errors: dict[str, str],
) -> str:
    """Load one-hour slices in a plotting process and save one PNG."""
    trange = tracers_summary_trange(date, start_hour)
    products: dict[str, xr.Dataset] = {}
    conjugate_products: dict[str, xr.Dataset] = {}
    errors = dict(product_errors)
    for key, path_text in cache_paths.items():
        if key == "orbit" or key.startswith("conjugate:"):
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        try:
            with xr.open_dataset(path) as cached:
                products[key] = _trim_datetime_dims(cached, trange).load()
        except Exception as exc:
            errors[key] = f"Cache read failed: {type(exc).__name__}: {exc}"

    for satellite in conjugate_satellites:
        path_text = cache_paths.get(f"conjugate:{satellite}")
        if path_text is None or not Path(path_text).exists():
            continue
        try:
            with xr.open_dataset(path_text) as cached:
                conjugate_products[satellite] = _trim_datetime_dims(
                    cached, trange
                ).load()
        except Exception as exc:
            conjugate_errors[satellite] = (
                f"Cache read failed: {type(exc).__name__}: {exc}"
            )

    orbit: xr.Dataset | None = None
    orbit_path_text = cache_paths.get("orbit")
    if orbit_path_text is not None and Path(orbit_path_text).exists():
        try:
            with xr.open_dataset(orbit_path_text) as cached:
                orbit = _trim_datetime_dims(cached, trange).load()
        except Exception as exc:
            orbit_error = f"Cache read failed: {type(exc).__name__}: {exc}"

    daily = DailySummaryData(
        probe=probe,
        date=date,
        bin_interval_seconds=bin_interval_seconds,
        products=products,
        product_errors=errors,
        orbit=orbit,
        orbit_error=orbit_error,
        cache_paths={key: Path(value) for key, value in cache_paths.items()},
        conjugate_products=conjugate_products,
        conjugate_errors=conjugate_errors,
        conjugate_satellites=conjugate_satellites,
    )
    _, path = make_one_hour_summary_plot(
        daily, start_hour, save=True, close=True
    )
    return str(path)


def generate_daily_summary_plots(
    date: str | pd.Timestamp,
    *,
    probe: int = 2,
    bin_interval_seconds: float | None = None,
    conjugate_satellites: str | Sequence[str] | None = None,
    force_download: bool = False,
    save_daily_cache: bool = True,
    download_workers: int = 3,
    http_timeout_seconds: float = 60.0,
    download_attempt_timeout_seconds: float = 300.0,
    conjugacy_attempt_timeout_seconds: float = 1800.0,
    conjugacy_workers: int = 4,
    conjugacy_chunk_size: int = 48,
    download_max_attempts: int = 3,
    download_retry_wait_seconds: float = 10.0,
    plot_workers: int = 3,
    skip_existing_plots: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Load one daily cache and generate all 24 one-hour summary PNGs.

    Product downloads use killable subprocesses with bounded retries.
    Plotting also uses processes because Matplotlib is not thread-safe.
    Parallel plot workers load only their one-hour slices from daily NetCDF.
    """
    plot_workers = int(plot_workers)
    if not 1 <= plot_workers <= 4:
        raise ValueError("plot_workers must be between 1 and 4.")
    day_text = _normalize_date(date).strftime("%Y-%m-%d")
    expected_paths = {
        hour: tracers_summary_plot_path(day_text, hour, probe=probe)
        for hour in range(24)
    }
    daily = load_daily_summary_data(
        day_text,
        probe=probe,
        bin_interval_seconds=bin_interval_seconds,
        conjugate_satellites=conjugate_satellites,
        force_download=force_download,
        save_daily_cache=save_daily_cache,
        download_workers=download_workers,
        http_timeout_seconds=http_timeout_seconds,
        download_attempt_timeout_seconds=download_attempt_timeout_seconds,
        conjugacy_attempt_timeout_seconds=conjugacy_attempt_timeout_seconds,
        conjugacy_workers=conjugacy_workers,
        conjugacy_chunk_size=conjugacy_chunk_size,
        download_max_attempts=download_max_attempts,
        download_retry_wait_seconds=download_retry_wait_seconds,
        verbose=verbose,
    )
    existing_cache_paths = [
        path for path in daily.cache_paths.values() if path.exists()
    ]
    newest_cache_mtime = max(
        (path.stat().st_mtime for path in existing_cache_paths),
        default=0.0,
    )
    pending_hours = []
    for hour, path in expected_paths.items():
        plot_is_current = (
            path.exists() and path.stat().st_mtime >= newest_cache_mtime
        )
        if not (skip_existing_plots and plot_is_current):
            pending_hours.append(hour)
    if verbose and len(pending_hours) < 24:
        print(f"plot  : skipped {24 - len(pending_hours)} existing PNG(s)")

    cache_paths = {key: str(path) for key, path in daily.cache_paths.items()}
    parallel_ready = (
        plot_workers > 1
        and save_daily_cache
        and all(
            key in cache_paths and Path(cache_paths[key]).exists()
            for key in daily.products
        )
        and (
            daily.orbit is None
            or (
                "orbit" in cache_paths
                and Path(cache_paths["orbit"]).exists()
            )
        )
        and all(
            f"conjugate:{key}" in cache_paths
            and Path(cache_paths[f"conjugate:{key}"]).exists()
            for key in daily.conjugate_products
        )
    )

    if parallel_ready and pending_hours:
        product_errors = dict(daily.product_errors)
        orbit_error = daily.orbit_error
        conjugate_keys = daily.conjugate_satellites
        conjugate_errors = dict(daily.conjugate_errors)
        daily.products.clear()
        daily.conjugate_products.clear()
        daily.orbit = None
        context_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        context = mp.get_context(context_name)
        with ProcessPoolExecutor(
            max_workers=min(plot_workers, len(pending_hours)),
            mp_context=context,
        ) as executor:
            future_hours = {
                executor.submit(
                    _plot_cached_hour,
                    daily.date,
                    probe,
                    daily.bin_interval_seconds,
                    hour,
                    cache_paths,
                    product_errors,
                    orbit_error,
                    conjugate_keys,
                    conjugate_errors,
                ): hour
                for hour in pending_hours
            }
            for future in as_completed(future_hours):
                hour = future_hours[future]
                path = Path(future.result())
                if verbose:
                    print(
                        f"plot  : {hour:02d}:00-{hour + 1:02d}:00 {path}"
                    )
    else:
        if verbose and plot_workers > 1 and pending_hours:
            print("plot  : parallel cache mode unavailable; using one process")
        for hour in pending_hours:
            _, path = make_one_hour_summary_plot(
                daily, hour, save=True, close=True
            )
            if verbose:
                print(f"plot  : {hour:02d}:00-{hour + 1:02d}:00 {path}")

    return [expected_paths[hour] for hour in range(24)]


def generate_summary_plots_for_date_range(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    probe: int = 2,
    bin_interval_seconds: float | None = None,
    conjugate_satellites: str | Sequence[str] | None = None,
    force_download: bool = False,
    save_daily_cache: bool = True,
    download_workers: int = 3,
    http_timeout_seconds: float = 60.0,
    download_attempt_timeout_seconds: float = 300.0,
    conjugacy_attempt_timeout_seconds: float = 1800.0,
    conjugacy_workers: int = 4,
    conjugacy_chunk_size: int = 48,
    download_max_attempts: int = 3,
    download_retry_wait_seconds: float = 10.0,
    plot_workers: int = 3,
    skip_existing_plots: bool = False,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Generate 24 one-hour plots per UTC day over an inclusive date range."""
    start = _normalize_date(start_date)
    end = _normalize_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    days = pd.date_range(start, end, freq="1D")
    all_paths: list[Path] = []
    failures: list[tuple[str, Exception]] = []
    for index, day in enumerate(days, start=1):
        day_text = day.strftime("%Y-%m-%d")
        if verbose:
            print(f"\n=== {day_text} UTC ({index}/{len(days)}) ===")
        try:
            all_paths.extend(
                generate_daily_summary_plots(
                    day_text,
                    probe=probe,
                    bin_interval_seconds=bin_interval_seconds,
                    conjugate_satellites=conjugate_satellites,
                    force_download=force_download,
                    save_daily_cache=save_daily_cache,
                    download_workers=download_workers,
                    http_timeout_seconds=http_timeout_seconds,
                    download_attempt_timeout_seconds=(
                        download_attempt_timeout_seconds
                    ),
                    conjugacy_attempt_timeout_seconds=(
                        conjugacy_attempt_timeout_seconds
                    ),
                    conjugacy_workers=conjugacy_workers,
                    conjugacy_chunk_size=conjugacy_chunk_size,
                    download_max_attempts=download_max_attempts,
                    download_retry_wait_seconds=(
                        download_retry_wait_seconds
                    ),
                    plot_workers=plot_workers,
                    skip_existing_plots=skip_existing_plots,
                    verbose=verbose,
                )
            )
        except Exception as exc:
            failures.append((day_text, exc))
            if not continue_on_error:
                raise
            if verbose:
                print(f"day   : FAILED {type(exc).__name__}: {exc}")
    if failures and verbose:
        print(f"\nfailed days: {len(failures)}/{len(days)}")
        for day_text, exc in failures:
            print(f"  {day_text}: {type(exc).__name__}: {exc}")
    return all_paths
