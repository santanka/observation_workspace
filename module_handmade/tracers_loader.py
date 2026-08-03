"""CDAWeb/xarray loader for NASA TRACERS mission data.

This module retrieves calibrated/public CDAWeb products without applying
instrument-specific quality masks or additional response calibration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr


SPEDAS_DATA_DIR = Path(os.environ.get("SPEDAS_DATA_DIR", "/mnt/j/observation_data"))
os.environ.setdefault("SPACEPY", str(SPEDAS_DATA_DIR / ".spacepy"))

from cdasws import CdasWs  # noqa: E402  (SPACEPY must be set before this import)
from cdasws.datarepresentation import DataRepresentation  # noqa: E402


TRACERS_DATA_DIR = Path(
    os.environ.get("TRACERS_DATA_DIR", SPEDAS_DATA_DIR / "tracers")
)
SSCWEB_HAPI_SERVER = "https://hapi-server.org/servers/SSCWeb/hapi"
DEFAULT_ORBIT_PARAMETERS = (
    "X_GEO",
    "Y_GEO",
    "Z_GEO",
    "Lat_GEO",
    "Lon_GEO",
    "Lat_GM",
    "LT_GM",
    "Radius",
    "DipInv",
)

KNOWN_DATASETS: dict[tuple[str, int, str], str] = {
    ("ace", 1, "def"): "TS1_L2_ACE_DEF",
    ("ace", 2, "def"): "TS2_L2_ACE_DEF",
    ("aci", 1, "ipd"): "TS1_L2_ACI_IPD",
    ("aci", 2, "ipd"): "TS2_L2_ACI_IPD",
    ("efi", 1, "eac"): "TS1_L2_EFI_EAC",
    ("efi", 1, "ehf"): "TS1_L2_EFI_EHF",
    ("efi", 1, "hsk"): "TS1_L2_EFI_HSK",
    ("efi", 1, "vdc"): "TS1_L2_EFI_VDC",
    ("efi", 2, "eac"): "TS2_L2_EFI_EAC",
    ("efi", 2, "ehf"): "TS2_L2_EFI_EHF",
    ("efi", 2, "hsk"): "TS2_L2_EFI_HSK",
    ("efi", 2, "vdc"): "TS2_L2_EFI_VDC",
    ("msc", 1, "bac"): "TS1_L2_MSC_BAC",
    ("msc", 2, "bac"): "TS2_L2_MSC_BAC",
    ("magic", 2, "bdc-16sps"): "TS2_L2_MAGIC_BDC-16SPS",
}

_DATASET_RE = re.compile(
    r"^TS(?P<probe>[12])_L(?P<level>\d+)_(?P<instrument>[A-Z0-9]+)_"
    r"(?P<datatype>[A-Z0-9-]+)$"
)
_SOURCE_VERSION_RE = re.compile(
    r"(?:^|_)v(?P<version>\d+(?:\.\d+)*)(?:\.cdf)?$", re.IGNORECASE
)

TimeLike = str | datetime | np.datetime64 | pd.Timestamp

__all__ = [
    "TRACERS_DATA_DIR",
    "SSCWEB_HAPI_SERVER",
    "DEFAULT_ORBIT_PARAMETERS",
    "KNOWN_DATASETS",
    "TracersLoadError",
    "DatasetNotAvailableError",
    "AmbiguousDatasetError",
    "TracersNoDataError",
    "TracersRequestError",
    "TracersSourceVersion",
    "find_tracers_datasets",
    "normalize_tracers_trange",
    "tracers_summary_paths",
    "tracers_summary_trange",
    "tracers_summary_plot_path",
    "tracers_daily_path",
    "tracers_orbit_daily_path",
    "get_tracers_inventory",
    "get_tracers_source_version",
    "get_tracers_variables",
    "resolve_tracers_dataset",
    "load_tracers_xarray",
    "load_tracers_orbit",
    "netcdf_safe_copy",
    "save_tracers_dataset",
    "default_tracers_path",
    "load_tracers",
    "ace",
    "aci",
    "efi",
    "msc",
    "magic",
]


class TracersLoadError(RuntimeError):
    """Base exception for this loader."""


class DatasetNotAvailableError(TracersLoadError):
    """The requested instrument/product is not registered at CDAWeb."""


class AmbiguousDatasetError(TracersLoadError):
    """More than one dataset matches a requested alias."""


class TracersNoDataError(TracersLoadError):
    """The dataset exists, but no data were returned in the interval."""


class TracersRequestError(TracersLoadError):
    """CDAWeb communication or server-side request failure."""


@dataclass(frozen=True)
class TracersSourceVersion:
    """Current original-file version information returned by CDAWeb."""

    dataset_id: str
    version: str
    files: tuple[str, ...]
    last_modified: tuple[str, ...]


def _canonical_version(value: object) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.lower().startswith("v"):
        text = text[1:]
    if not re.fullmatch(r"\d+(?:\.\d+)*", text):
        raise ValueError(f"Invalid source version: {value!r}.")
    return "v" + text


def _version_key(value: object) -> tuple[int, ...]:
    canonical = _canonical_version(value)[1:]
    parts = [int(item) for item in canonical.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _validate_dataset_id(dataset_id: str) -> str:
    normalized = str(dataset_id).strip().upper()
    if _DATASET_RE.fullmatch(normalized) is None:
        raise ValueError(f"Invalid TRACERS dataset ID: {dataset_id!r}")
    return normalized


def _normalize_trange(trange: Sequence[TimeLike]) -> tuple[str, str]:
    if len(trange) != 2:
        raise ValueError("trange must contain exactly [start, stop].")

    normalized: list[pd.Timestamp] = []
    for value in trange:
        try:
            timestamp = pd.Timestamp(value)
        except Exception as exc:
            raise ValueError(f"Invalid time value: {value!r}") from exc
        if pd.isna(timestamp):
            raise ValueError(f"Invalid time value: {value!r}")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        normalized.append(timestamp)

    start, stop = normalized
    if start >= stop:
        raise ValueError(f"trange must satisfy start < stop: {start} >= {stop}")
    return tuple(t.isoformat().replace("+00:00", "Z") for t in (start, stop))


def normalize_tracers_trange(
    trange: Sequence[TimeLike],
    *,
    max_duration: str | pd.Timedelta | None = None,
) -> tuple[str, str]:
    """Normalize a time range to UTC and optionally enforce a maximum width."""
    start, stop = _normalize_trange(trange)
    if max_duration is not None:
        limit = pd.Timedelta(max_duration)
        if limit <= pd.Timedelta(0):
            raise ValueError("max_duration must be positive.")
        duration = pd.Timestamp(stop) - pd.Timestamp(start)
        if duration > limit:
            raise ValueError(
                f"TRANGE duration {duration} exceeds the allowed maximum {limit}."
            )
    return start, stop


def tracers_summary_paths(
    trange: Sequence[TimeLike],
    *,
    probe: int = 2,
    output_dir: str | Path = TRACERS_DATA_DIR / "summary",
    max_duration: str | pd.Timedelta = "1D",
) -> tuple[Path, Path]:
    """Return the cache directory and PNG path for a summary interval."""
    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    start, stop = normalize_tracers_trange(trange, max_duration=max_duration)
    start_tag = pd.Timestamp(start).strftime("%Y%m%dT%H%M%S")
    stop_tag = pd.Timestamp(stop).strftime("%Y%m%dT%H%M%S")
    range_tag = f"{start_tag}_{stop_tag}"
    cache_dir = Path(output_dir).expanduser() / range_tag
    figure_path = cache_dir / f"tracers{probe}_summary_{range_tag}.png"
    return cache_dir, figure_path


def _normalize_utc_date(date: TimeLike) -> pd.Timestamp:
    timestamp = pd.Timestamp(date)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid UTC date: {date!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def tracers_summary_trange(
    date: TimeLike,
    start_hour_utc: int,
) -> tuple[str, str]:
    """Build one of the 24 non-overlapping one-hour UTC summary slots."""
    start_hour_utc = int(start_hour_utc)
    if start_hour_utc not in range(24):
        raise ValueError("start_hour_utc must be an integer from 0 through 23.")
    day = _normalize_utc_date(date)
    start = day + pd.Timedelta(hours=start_hour_utc)
    stop = start + pd.Timedelta(hours=1)
    return tuple(
        value.isoformat().replace("+00:00", "Z") for value in (start, stop)
    )


def tracers_summary_plot_path(
    date: TimeLike,
    start_hour_utc: int,
    *,
    probe: int = 2,
    output_dir: str | Path = TRACERS_DATA_DIR / "summary_plot",
) -> Path:
    """Return a probe/year/month summary PNG path for a one-hour slot."""
    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    start, stop = tracers_summary_trange(date, start_hour_utc)
    day = pd.Timestamp(start)
    start_tag = day.strftime("%Y%m%d_%H%M")
    stop_tag = f"{int(start_hour_utc) + 1:02d}00"
    filename = f"tracers{probe}_summary_{start_tag}_{stop_tag}.png"
    return (
        Path(output_dir).expanduser()
        / f"ts{probe}"
        / day.strftime("%Y")
        / day.strftime("%m")
        / filename
    )


def tracers_daily_path(
    dataset_id: str,
    date: TimeLike,
    *,
    bin_interval_seconds: float | None = 60.0,
    source_version: str | None = None,
    output_dir: str | Path = TRACERS_DATA_DIR,
) -> Path:
    """Return an ERG-like daily cache path.

    ``bin_interval_seconds=None`` selects a native-cadence cache and gives it
    a distinct ``_native.nc`` suffix.  Numeric values retain the historical
    binned-cache naming scheme.
    """
    dataset_id = _validate_dataset_id(dataset_id)
    parsed = _DATASET_RE.fullmatch(dataset_id)
    assert parsed is not None
    fields = parsed.groupdict()
    day = _normalize_utc_date(date)
    if bin_interval_seconds is None:
        bin_tag = "native"
    else:
        interval = float(bin_interval_seconds)
        if interval <= 0:
            raise ValueError("bin_interval_seconds must be positive or None.")
        bin_tag = f"{interval:g}s".replace(".", "p")
    version_tag = ""
    if source_version is not None:
        version = _canonical_version(source_version)
        version_tag = f"_{version}"
    filename = (
        f"{dataset_id.lower()}_{day.strftime('%Y%m%d')}"
        f"{version_tag}_{bin_tag}.nc"
    )
    return (
        Path(output_dir).expanduser()
        / f"ts{fields['probe']}"
        / fields["instrument"].lower()
        / f"l{fields['level']}"
        / fields["datatype"].lower()
        / day.strftime("%Y")
        / day.strftime("%m")
        / filename
    )


def tracers_orbit_daily_path(
    probe: int,
    date: TimeLike,
    *,
    output_dir: str | Path = TRACERS_DATA_DIR,
) -> Path:
    """Return the SSCWeb 60 s daily orbit cache path."""
    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    day = _normalize_utc_date(date)
    filename = f"tracers{probe}_orbit_sscweb_{day.strftime('%Y%m%d')}_60s.nc"
    return (
        Path(output_dir).expanduser()
        / f"ts{probe}"
        / "orbit"
        / "sscweb"
        / day.strftime("%Y")
        / day.strftime("%m")
        / filename
    )


def _status_code(status: object) -> int | None:
    if isinstance(status, (int, np.integer)):
        return int(status)
    if isinstance(status, Mapping):
        http = status.get("http")
        if isinstance(http, Mapping) and http.get("status_code") is not None:
            return int(http["status_code"])
    return None


@lru_cache(maxsize=1)
def _discover_tracers_datasets() -> tuple[str, ...]:
    from pyspedas import find_datasets

    try:
        found = find_datasets(mission="TRACERS", label=False, quiet=True)
    except Exception as exc:
        raise TracersRequestError("TRACERS dataset discovery failed.") from exc
    dataset_ids = sorted({str(item).strip().upper() for item in found})
    return tuple(item for item in dataset_ids if _DATASET_RE.fullmatch(item))


def find_tracers_datasets(
    instrument: str | None = None,
    *,
    probe: int | None = None,
    datatype: str | None = None,
    refresh: bool = False,
) -> list[str]:
    """Query CDAWeb and optionally filter current TRACERS dataset IDs."""
    if probe is not None and int(probe) not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    if refresh:
        _discover_tracers_datasets.cache_clear()

    instrument_norm = instrument.upper() if instrument else None
    datatype_norm = datatype.upper().replace("_", "-") if datatype else None
    matches: list[str] = []
    for dataset_id in _discover_tracers_datasets():
        parsed = _DATASET_RE.fullmatch(dataset_id)
        if parsed is None:
            continue
        fields = parsed.groupdict()
        if instrument_norm and fields["instrument"] != instrument_norm:
            continue
        if probe is not None and int(fields["probe"]) != int(probe):
            continue
        if datatype_norm and fields["datatype"] != datatype_norm:
            continue
        matches.append(dataset_id)
    return matches


def get_tracers_inventory(
    dataset_id: str,
    *,
    client: CdasWs | None = None,
    request_timeout: float = 60.0,
) -> pd.DataFrame:
    """Return CDAWeb availability intervals as a UTC pandas DataFrame."""
    dataset_id = _validate_dataset_id(dataset_id)
    if float(request_timeout) <= 0:
        raise ValueError("request_timeout must be positive.")
    cdas = (
        client
        if client is not None
        else CdasWs(timeout=float(request_timeout), disable_cache=True)
    )
    try:
        intervals = cdas.get_inventory(dataset_id)
    except requests.exceptions.RequestException as exc:
        raise TracersRequestError(f"Inventory request failed for {dataset_id}.") from exc
    rows = [
        {
            "dataset_id": dataset_id,
            "start": pd.Timestamp(interval.start).tz_convert("UTC"),
            "stop": pd.Timestamp(interval.end).tz_convert("UTC"),
        }
        for interval in intervals
    ]
    result = pd.DataFrame(rows, columns=["dataset_id", "start", "stop"])
    if not result.empty:
        result["duration"] = result["stop"] - result["start"]
    return result


def get_tracers_source_version(
    dataset_id: str,
    trange: Sequence[TimeLike],
    *,
    client: CdasWs | None = None,
    request_timeout: float = 60.0,
    max_attempts: int = 3,
    retry_wait: float = 1.0,
) -> TracersSourceVersion:
    """Resolve the current original-CDF version for an interval.

    CDAWeb's original-file endpoint is used because the inventory endpoint
    contains availability intervals but no file version.  All source file
    names and modification timestamps are retained in the returned manifest.
    """
    dataset_id = _validate_dataset_id(dataset_id)
    start, stop = _normalize_trange(trange)
    if float(request_timeout) <= 0:
        raise ValueError("request_timeout must be positive.")
    if not 1 <= int(max_attempts) <= 5:
        raise ValueError("max_attempts must be between 1 and 5.")
    last_error: Exception | None = None
    result: object = None
    status: object = None
    for attempt in range(int(max_attempts)):
        cdas = (
            client
            if client is not None
            else CdasWs(timeout=float(request_timeout), disable_cache=True)
        )
        try:
            status, result = cdas.get_original_files(dataset_id, start, stop)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < int(max_attempts):
                time.sleep(float(retry_wait) * 2**attempt)
                continue
            raise TracersRequestError(
                f"Original-file query failed for {dataset_id} in "
                f"[{start}, {stop}]."
            ) from exc
        status_code = _status_code(status)
        if (
            status_code in {429, 502, 503, 504}
            and attempt + 1 < int(max_attempts)
        ):
            time.sleep(float(retry_wait) * 2**attempt)
            continue
        break
    if _status_code(status) != 200 or not isinstance(result, list):
        raise TracersRequestError(
            f"CDAWeb original-file query returned status={status!r} "
            f"for {dataset_id}."
        ) from last_error
    if not result:
        raise TracersNoDataError(
            f"No original files for {dataset_id} in [{start}, {stop}]."
        )

    files: list[str] = []
    modified: list[str] = []
    versions: list[str] = []
    for item in result:
        if not isinstance(item, Mapping) or not item.get("Name"):
            continue
        name = str(item["Name"])
        files.append(name)
        modified.append(str(item.get("LastModified", "")))
        filename = Path(name).name
        match = _SOURCE_VERSION_RE.search(filename)
        if match is None:
            raise TracersRequestError(
                f"Could not parse a source version from CDAWeb file {filename!r}."
            )
        versions.append(_canonical_version(match.group("version")))
    if not files:
        raise TracersNoDataError(
            f"CDAWeb returned no named original files for {dataset_id}."
        )
    unique_versions = sorted(set(versions), key=_version_key)
    if len(unique_versions) != 1:
        raise TracersRequestError(
            f"Mixed source versions for {dataset_id} in [{start}, {stop}]: "
            f"{unique_versions}. Split the request before caching."
        )
    return TracersSourceVersion(
        dataset_id=dataset_id,
        version=unique_versions[0],
        files=tuple(files),
        last_modified=tuple(modified),
    )


def get_tracers_variables(
    dataset_id: str, *, client: CdasWs | None = None
) -> pd.DataFrame:
    """Return CDAWeb variable names and descriptions for one dataset."""
    dataset_id = _validate_dataset_id(dataset_id)
    cdas = client if client is not None else CdasWs()
    try:
        variables = cdas.get_variables(dataset_id)
    except requests.exceptions.RequestException as exc:
        raise TracersRequestError(f"Variable request failed for {dataset_id}.") from exc
    result = pd.DataFrame(variables)
    if not result.empty:
        result.insert(0, "DatasetId", dataset_id)
    return result


def resolve_tracers_dataset(instrument: str, probe: int, datatype: str) -> str:
    """Resolve an instrument alias without silently guessing a product."""
    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    key = (instrument.lower(), probe, datatype.lower().replace("_", "-"))
    if key in KNOWN_DATASETS:
        return KNOWN_DATASETS[key]

    candidates = find_tracers_datasets(
        instrument=instrument, probe=probe, datatype=datatype
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise AmbiguousDatasetError(
            f"Multiple datasets match instrument={instrument!r}, probe={probe}, "
            f"datatype={datatype!r}: {candidates}"
        )
    raise DatasetNotAvailableError(
        f"No CDAWeb dataset matches instrument={instrument!r}, probe={probe}, "
        f"datatype={datatype!r}. Run find_tracers_datasets(refresh=True)."
    )


def load_tracers_xarray(
    dataset_id: str,
    trange: Sequence[TimeLike],
    *,
    variables: Sequence[str] | None = None,
    bin_data: Mapping[str, Any] | None = None,
    client: CdasWs | None = None,
    max_attempts: int = 3,
    retry_wait: float = 1.0,
    request_timeout: float = 120.0,
) -> xr.Dataset:
    """Load one CDAWeb dataset explicitly as an xarray.Dataset.

    ``bin_data`` is passed to CDAS as ``binData``. It is useful for a
    low-cadence overview but changes the temporal averaging and is recorded
    in the returned metadata.
    """
    dataset_id = _validate_dataset_id(dataset_id)
    start, stop = _normalize_trange(trange)
    requested_variables = (
        list(variables) if variables is not None else ["ALL-VARIABLES"]
    )
    if not requested_variables or any(not str(name).strip() for name in requested_variables):
        raise ValueError("variables must contain at least one non-empty name.")
    if not 1 <= int(max_attempts) <= 5:
        raise ValueError("max_attempts must be between 1 and 5.")
    if float(retry_wait) < 0:
        raise ValueError("retry_wait must be non-negative.")
    if float(request_timeout) <= 0:
        raise ValueError("request_timeout must be positive.")

    retryable_statuses = {429, 502, 503, 504}
    status: object = None
    ds: object = None
    attempts_used = 0
    for attempt in range(int(max_attempts)):
        attempts_used = attempt + 1
        cdas = (
            client
            if client is not None
            else CdasWs(timeout=float(request_timeout), disable_cache=True)
        )
        keywords: dict[str, Any] = {
            "dataRepresentation": DataRepresentation.XARRAY,
        }
        if bin_data:
            keywords["binData"] = dict(bin_data)
        try:
            status, ds = cdas.get_data(
                dataset_id,
                requested_variables,
                start,
                stop,
                **keywords,
            )
        except requests.exceptions.RequestException as exc:
            if attempts_used >= int(max_attempts):
                raise TracersRequestError(
                    f"CDAWeb request failed after {attempts_used} attempt(s) for "
                    f"{dataset_id} in [{start}, {stop}]."
                ) from exc
            time.sleep(float(retry_wait) * 2**attempt)
            continue

        code = _status_code(status)
        if code in retryable_statuses and attempts_used < int(max_attempts):
            time.sleep(float(retry_wait) * 2**attempt)
            continue
        break

    code = _status_code(status)
    if code == 204 or (code == 200 and ds is None):
        raise TracersNoDataError(
            f"No data for {dataset_id} in [{start}, {stop}]; status={status!r}"
        )
    if code != 200:
        raise TracersRequestError(
            f"CDAWeb returned HTTP status {code!r} for {dataset_id}: {status!r}"
        )
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected xarray.Dataset, got {type(ds)!r}.")

    ds.attrs.update(
        {
            "tracers_dataset_id": dataset_id,
            "tracers_loader_backend": "cdasws-xarray",
            "tracers_request_start_utc": start,
            "tracers_request_stop_utc": stop,
            "tracers_requested_variables": json.dumps(requested_variables),
            "tracers_bin_data": json.dumps(dict(bin_data or {}), sort_keys=True),
            "tracers_request_attempts": attempts_used,
            "tracers_retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "tracers_calibration_applied": "none by this loader",
            "tracers_quality_mask_applied": "none by this loader",
        }
    )
    return ds


def load_tracers_orbit(
    trange: Sequence[TimeLike],
    *,
    probe: int = 2,
    parameters: Sequence[str] = DEFAULT_ORBIT_PARAMETERS,
    server: str = SSCWEB_HAPI_SERVER,
    max_duration: str | pd.Timedelta | None = "1D",
    max_attempts: int = 3,
    retry_wait: float = 1.0,
) -> xr.Dataset:
    """Load 60 s TRACERS ephemeris from SSCWeb HAPI as xarray.

    The SSC observatory identifiers are ``tracers1`` and ``tracers2``.
    Default positions are geographic Earth-fixed coordinates in Earth radii.
    No interpolation is applied by this function.
    """
    from hapiclient import hapi

    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    if not parameters or any(not str(name).strip() for name in parameters):
        raise ValueError("parameters must contain at least one non-empty name.")
    if not 1 <= int(max_attempts) <= 5:
        raise ValueError("max_attempts must be between 1 and 5.")
    if float(retry_wait) < 0:
        raise ValueError("retry_wait must be non-negative.")

    start, stop = normalize_tracers_trange(trange, max_duration=max_duration)
    ssc_id = f"tracers{probe}"
    parameter_names = [str(name).strip() for name in parameters]
    last_error: Exception | None = None
    data: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None
    for attempt in range(int(max_attempts)):
        try:
            data, metadata = hapi(
                server,
                ssc_id,
                ",".join(parameter_names),
                start,
                stop,
                logging=False,
                usecache=False,
                cache=False,
                format="csv",
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= int(max_attempts):
                raise TracersRequestError(
                    f"SSCWeb orbit request failed after {attempt + 1} attempt(s) "
                    f"for {ssc_id} in [{start}, {stop}]."
                ) from exc
            time.sleep(float(retry_wait) * 2**attempt)

    if data is None or metadata is None or len(data) == 0:
        raise TracersNoDataError(
            f"No SSCWeb orbit data for {ssc_id} in [{start}, {stop}]."
        ) from last_error
    if "Time" not in data.dtype.names:
        raise TypeError("SSCWeb HAPI response has no Time field.")

    raw_time = data["Time"]
    if raw_time.dtype.kind == "S":
        time_text = np.char.decode(raw_time, "utf-8")
    else:
        time_text = raw_time.astype(str)
    time_index = pd.to_datetime(
        time_text,
        format="%Y-%jT%H:%M:%SZ",
        utc=True,
        errors="raise",
    )
    time_values = time_index.tz_convert(None).to_numpy(dtype="datetime64[ns]")

    parameter_metadata = {
        str(item.get("name")): item
        for item in metadata.get("parameters", [])
        if isinstance(item, Mapping) and item.get("name") is not None
    }
    data_vars: dict[str, tuple[str, np.ndarray, dict[str, Any]]] = {}
    for name in parameter_names:
        if name not in data.dtype.names:
            raise KeyError(f"SSCWeb response omitted requested parameter {name!r}.")
        values = np.asarray(data[name])
        if values.dtype.kind in "iufc":
            values = values.astype(float, copy=False)
            values = np.where(np.abs(values) >= 1.0e30, np.nan, values)
        item = parameter_metadata.get(name, {})
        attrs = {
            "units": str(item.get("units", "")),
            "long_name": str(item.get("description", name)),
            "ssc_fill_value": str(item.get("fill", "")),
        }
        data_vars[name] = ("time", values, attrs)

    ds = xr.Dataset(data_vars, coords={"time": time_values})
    ds["time"].attrs.update({"standard_name": "time", "timezone": "UTC"})
    if "Radius" in ds:
        ds["Altitude"] = (ds["Radius"] - 1.0) * 6371.2
        ds["Altitude"].attrs.update(
            {
                "units": "km",
                "long_name": "Spherical geocentric altitude",
                "derivation": "(SSCWeb Radius - 1) * 6371.2 km",
                "earth_radius_km": 6371.2,
            }
        )
    if "LT_GM" in ds:
        mlt = np.full(ds.sizes["time"], np.nan, dtype=float)
        for index, value in enumerate(np.asarray(ds["LT_GM"].values).astype(str)):
            try:
                hour, minute, second = (float(part) for part in value.split(":"))
            except (TypeError, ValueError):
                continue
            if 0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60:
                mlt[index] = hour + minute / 60.0 + second / 3600.0
        ds["MLT"] = ("time", mlt)
        ds["MLT"].attrs.update(
            {
                "units": "hour",
                "long_name": "SSCWeb geomagnetic local time",
                "source_parameter": "LT_GM",
                "definition_note": "SSCWeb LT_GM; not recomputed with AACGM",
            }
        )
    if "DipInv" in ds:
        ds["InvariantLatitude"] = ds["DipInv"].copy(deep=False)
        ds["InvariantLatitude"].attrs.update(
            {
                "units": "deg",
                "long_name": "SSCWeb unsigned dipole invariant latitude",
                "source_parameter": "DipInv",
                "model": "SSCWeb metadata: IGRF + Tsyganenko 89C",
                "hemisphere_information": "none; SSCWeb DipInv is non-negative",
            }
        )
    if "DipInv" in ds and "Lat_GM" in ds:
        signed_invlat = xr.where(
            ds["Lat_GM"] < 0.0,
            -np.abs(ds["DipInv"]),
            np.abs(ds["DipInv"]),
        )
        ds["SignedInvariantLatitude"] = signed_invlat
        ds["SignedInvariantLatitude"].attrs.update(
            {
                "units": "deg",
                "long_name": "Hemisphere-signed SSCWeb dipole invariant latitude",
                "source_parameter": "DipInv",
                "hemisphere_source_parameter": "Lat_GM",
                "derivation": "abs(DipInv) with sign from SSCWeb Lat_GM",
                "model": "SSCWeb metadata: IGRF + Tsyganenko 89C",
                "equator_convention": "positive when Lat_GM equals zero",
            }
        )
    ds.attrs.update(
        {
            "tracers_orbit_source": "SSCWeb HAPI",
            "tracers_orbit_server": server,
            "tracers_ssc_id": ssc_id,
            "tracers_probe": probe,
            "tracers_request_start_utc": start,
            "tracers_request_stop_utc": stop,
            "tracers_requested_parameters": json.dumps(parameter_names),
            "tracers_orbit_cadence": str(metadata.get("cadence", "")),
            "tracers_retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "tracers_orbit_interpolation": "none",
        }
    )
    return ds


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if value is None or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _netcdf_attr(value: Any) -> str | int | float:
    simple = _jsonable(value)
    if simple is None:
        return "null"
    if isinstance(simple, (str, int, float)):
        return simple
    return json.dumps(simple, ensure_ascii=False, sort_keys=True)


def netcdf_safe_copy(ds: xr.Dataset) -> xr.Dataset:
    """Copy metadata only and make all attributes NetCDF4-compatible."""
    safe = ds.copy(deep=False)
    safe.attrs = {str(key): _netcdf_attr(value) for key, value in ds.attrs.items()}
    for name in safe.variables:
        safe[name].attrs = {
            str(key): _netcdf_attr(value) for key, value in ds[name].attrs.items()
        }
        if safe[name].dtype.kind in "Mm":
            for key in ("units", "calendar"):
                if key in safe[name].attrs:
                    safe[name].attrs[f"cdf_{key}"] = safe[name].attrs.pop(key)
    return safe


def save_tracers_dataset(
    ds: xr.Dataset,
    path: str | Path,
    *,
    overwrite: bool = False,
    complevel: int = 4,
) -> Path:
    """Atomically save a TRACERS xarray Dataset as compressed NetCDF4."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"ds must be xarray.Dataset, got {type(ds)!r}.")
    if not 0 <= int(complevel) <= 9:
        raise ValueError("complevel must be between 0 and 9.")

    output = Path(path).expanduser()
    if output.suffix == "":
        output = output.with_suffix(".nc")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    safe = netcdf_safe_copy(ds)
    encoding = {
        name: {"zlib": True, "complevel": int(complevel)}
        for name, array in safe.data_vars.items()
        if array.ndim > 0 and array.dtype.kind not in "OUSV"
    }
    temporary = output.with_name(f".{output.name}.part")
    try:
        safe.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def default_tracers_path(
    dataset_id: str,
    trange: Sequence[TimeLike],
    *,
    source_version: str | None = None,
    output_dir: str | Path = TRACERS_DATA_DIR,
) -> Path:
    """Build a deterministic cache path from dataset ID and UTC range."""
    dataset_id = _validate_dataset_id(dataset_id)
    start, stop = _normalize_trange(trange)
    start_tag = pd.Timestamp(start).strftime("%Y%m%dT%H%M%S")
    stop_tag = pd.Timestamp(stop).strftime("%Y%m%dT%H%M%S")
    version_tag = (
        "" if source_version is None else f"_{_canonical_version(source_version)}"
    )
    filename = (
        f"{dataset_id.lower()}_{start_tag}_{stop_tag}{version_tag}.nc"
    )
    return Path(output_dir) / dataset_id / filename


def load_tracers(
    instrument: str,
    trange: Sequence[TimeLike],
    *,
    probe: int,
    datatype: str,
    variables: Sequence[str] | None = None,
    bin_data: Mapping[str, Any] | None = None,
    save_path: str | Path | None = None,
    overwrite: bool = False,
    client: CdasWs | None = None,
    max_attempts: int = 3,
    retry_wait: float = 1.0,
) -> xr.Dataset:
    """Resolve an instrument product, load it, and optionally save it."""
    dataset_id = resolve_tracers_dataset(instrument, probe, datatype)
    ds = load_tracers_xarray(
        dataset_id,
        trange,
        variables=variables,
        bin_data=bin_data,
        client=client,
        max_attempts=max_attempts,
        retry_wait=retry_wait,
    )
    if save_path is not None:
        saved = save_tracers_dataset(ds, save_path, overwrite=overwrite)
        ds.attrs["tracers_saved_path"] = str(saved.resolve())
    return ds


def ace(
    trange: Sequence[TimeLike], *, probe: int = 1, datatype: str = "def", **kwargs: Any
) -> xr.Dataset:
    return load_tracers("ace", trange, probe=probe, datatype=datatype, **kwargs)


def aci(
    trange: Sequence[TimeLike], *, probe: int = 1, datatype: str = "ipd", **kwargs: Any
) -> xr.Dataset:
    return load_tracers("aci", trange, probe=probe, datatype=datatype, **kwargs)


def efi(
    trange: Sequence[TimeLike], *, probe: int = 1, datatype: str = "eac", **kwargs: Any
) -> xr.Dataset:
    return load_tracers("efi", trange, probe=probe, datatype=datatype, **kwargs)


def msc(
    trange: Sequence[TimeLike], *, probe: int = 1, datatype: str = "bac", **kwargs: Any
) -> xr.Dataset:
    return load_tracers("msc", trange, probe=probe, datatype=datatype, **kwargs)


def magic(
    trange: Sequence[TimeLike],
    *,
    probe: int = 2,
    datatype: str = "bdc-16sps",
    **kwargs: Any,
) -> xr.Dataset:
    return load_tracers("magic", trange, probe=probe, datatype=datatype, **kwargs)
