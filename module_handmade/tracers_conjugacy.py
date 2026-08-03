"""T04 magnetic-conjugacy products for TRACERS summary plots."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .tracers_loader import TRACERS_DATA_DIR, save_tracers_dataset
from .tracers_t04_worker import (
    initialize_t04_worker as _initialize_t04_worker,
    trace_arase_chunk as _trace_arase_chunk,
)


EARTH_RADIUS_KM = 6371.2
T04_COLUMNS = (
    "Pdyn",
    "Dst",
    "ByIMF",
    "BzIMF",
    "W1",
    "W2",
    "W3",
    "W4",
    "W5",
    "W6",
)

@dataclass(frozen=True)
class ConjugateSatelliteSpec:
    key: str
    label: str
    color: str


CONJUGATE_SATELLITE_REGISTRY: dict[str, ConjugateSatelliteSpec] = {
    "Arase": ConjugateSatelliteSpec(
        key="Arase",
        label="Arase T04 footprint",
        color="tab:green",
    ),
}


def normalize_conjugate_satellites(
    satellites: str | Sequence[str] | None,
) -> tuple[str, ...]:
    """Return unique canonical registry keys in user-specified order."""
    if satellites is None:
        return ()
    values = [satellites] if isinstance(satellites, str) else list(satellites)
    lookup = {key.casefold(): key for key in CONJUGATE_SATELLITE_REGISTRY}
    normalized: list[str] = []
    for value in values:
        requested = str(value).strip()
        if not requested:
            raise ValueError("Conjugate satellite keys must not be empty.")
        try:
            canonical = lookup[requested.casefold()]
        except KeyError as exc:
            known = ", ".join(CONJUGATE_SATELLITE_REGISTRY)
            raise ValueError(
                f"Unknown conjugate satellite {requested!r}; available: {known}."
            ) from exc
        if canonical not in normalized:
            normalized.append(canonical)
    return tuple(normalized)


def conjugacy_daily_path(
    satellite: str,
    probe: int,
    date: str | pd.Timestamp,
    *,
    output_dir: str | Path = TRACERS_DATA_DIR,
) -> Path:
    """Return the daily T04 footprint-cache path for one target satellite."""
    satellite = normalize_conjugate_satellites([satellite])[0]
    probe = int(probe)
    if probe not in (1, 2):
        raise ValueError("probe must be 1 or 2.")
    day = pd.Timestamp(date)
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    day = day.normalize()
    filename = (
        f"{satellite.lower()}_t04_footprint_at_tracers{probe}_"
        f"{day.strftime('%Y%m%d')}.nc"
    )
    return (
        Path(output_dir).expanduser()
        / "conjugacy"
        / f"ts{probe}"
        / satellite.lower()
        / "t04"
        / day.strftime("%Y")
        / day.strftime("%m")
        / filename
    )


def ts05_w_yearly_path(
    year: int, *, output_dir: str | Path = TRACERS_DATA_DIR
) -> Path:
    """Return the local yearly W1--W6 driver cache path."""
    year = int(year)
    return (
        Path(output_dir).expanduser()
        / "conjugacy"
        / "drivers"
        / "ts05_w"
        / f"{year:04d}"
        / f"ts05_w_{year:04d}.nc"
    )


def t04_driver_daily_path(
    date: str | pd.Timestamp,
    *,
    output_dir: str | Path = TRACERS_DATA_DIR,
) -> Path:
    """Return the daily cached OMNI/Dst input path used by T04."""
    day = pd.Timestamp(date)
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    day = day.normalize()
    return (
        Path(output_dir).expanduser()
        / "conjugacy"
        / "drivers"
        / "t04_omni"
        / day.strftime("%Y")
        / day.strftime("%m")
        / f"t04_omni_{day.strftime('%Y%m%d')}.nc"
    )


def _load_local_geopack() -> Any:
    """Load the workspace geopack-vectorize package despite namespace clashes."""
    package_root = Path(__file__).resolve().parents[1] / "geopack-vectorize"
    if not (package_root / "geopack" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Workspace geopack-vectorize package not found at {package_root}."
        )
    root_text = str(package_root)
    sys.path[:] = [item for item in sys.path if item != root_text]
    sys.path.insert(0, root_text)
    for name in tuple(sys.modules):
        if name == "geopack" or name.startswith("geopack."):
            del sys.modules[name]
    importlib.invalidate_caches()
    geopack = importlib.import_module("geopack")
    required = {
        "trace_vectorized",
        "smgsm_vectorized",
        "geogsm_vectorized",
        "recalc",
    }
    missing = sorted(required - set(dir(geopack)))
    if missing:
        raise ImportError(
            f"geopack-vectorize APIs are unavailable: missing={missing}."
        )
    return geopack


def _as_data_array(value: Any, name: str) -> xr.DataArray:
    """Convert a PySPEDAS tplot variable to a time-indexed DataArray."""
    if isinstance(value, xr.DataArray):
        da = value
    else:
        times = getattr(value, "times", None)
        data = getattr(value, "y", None)
        if times is None or data is None:
            raise TypeError(f"Could not interpret tplot variable {name!r}.")
        array = np.asarray(data)
        dims = ("time",) + tuple(f"dim{index}" for index in range(1, array.ndim))
        da = xr.DataArray(
            array,
            dims=dims,
            coords={"time": pd.to_datetime(np.asarray(times), unit="s")},
            name=name,
        )
    if "time" not in da.coords:
        first_dim = da.dims[0]
        if first_dim not in da.coords:
            raise ValueError(f"{name!r} has no time coordinate.")
        da = da.rename({first_dim: "time"})
    da = da.assign_coords(time=pd.to_datetime(da.time.values).to_numpy())
    _, unique = np.unique(da.time.values, return_index=True)
    return da.isel(time=np.sort(unique)).sortby("time")


def _nearest_values(
    da: xr.DataArray,
    target_times: np.ndarray,
    *,
    tolerance: str,
) -> np.ndarray:
    target = xr.DataArray(
        np.asarray(target_times, dtype="datetime64[ns]"),
        dims="time",
        name="time",
    )
    selected = da.reindex(
        time=target.values,
        method="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    return np.asarray(selected.values, dtype=float)


def _load_or_create_w_year(year: int) -> xr.Dataset:
    path = ts05_w_yearly_path(year)
    if path.exists():
        with xr.open_dataset(path) as cached:
            return cached.load()

    from pyspedas.geopack.get_w_params import get_w

    trange = [
        f"{year:04d}-01-01/00:00:00",
        f"{year:04d}-12-31/23:59:59",
    ]
    result = get_w(trange=trange, create_tvar=False)
    if not isinstance(result, dict) or "times" not in result:
        raise RuntimeError(f"TS05 W1--W6 data are unavailable for {year}.")
    times = pd.to_datetime(np.asarray(result["times"]), unit="s")
    ds = xr.Dataset(
        {
            name: ("time", np.asarray(result[name.lower()], dtype=float))
            for name in T04_COLUMNS[4:]
        },
        coords={"time": times.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "source": "PySPEDAS get_w / TS05_data_and_stuff",
            "model_usage": "T04 external field model W1-W6 drivers",
            "year": int(year),
        },
    )
    save_tracers_dataset(ds, path, overwrite=True)
    return ds


def _load_arase_position_gsm(
    trange: list[str], target_times: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], str]:
    import pyspedas as psp

    source_files = psp.projects.erg.orb(
        trange=trange,
        level="l2",
        datatype="def",
        downloadonly=True,
        ror=False,
    )
    if not source_files:
        raise RuntimeError("No Arase L2 orbit source CDF was resolved.")
    source_files = tuple(str(Path(item).resolve()) for item in source_files)
    versions = set()
    for item in source_files:
        match = re.search(r"_v(\d+(?:\.\d+)*)\.cdf$", Path(item).name, re.I)
        if match is None:
            raise RuntimeError(
                f"Could not parse Arase orbit version from {Path(item).name!r}."
            )
        versions.add("v" + match.group(1))
    if len(versions) != 1:
        raise RuntimeError(f"Mixed Arase orbit source versions: {sorted(versions)}.")
    source_version = next(iter(versions))

    psp.projects.erg.orb(
        trange=trange,
        level="l2",
        datatype="def",
        time_clip=True,
        no_update=True,
        ror=False,
    )
    raw = psp.get_data("erg_orb_l2_pos_gsm", xarray=True)
    if raw is None:
        raise RuntimeError("Arase orbit variable erg_orb_l2_pos_gsm was not loaded.")
    da = _as_data_array(raw, "erg_orb_l2_pos_gsm")
    if da.ndim != 2 or da.shape[1] != 3:
        raise ValueError(f"Unexpected Arase GSM position shape: {da.shape}.")
    values = np.asarray(da.values, dtype=float)
    radius_median = float(np.nanmedian(np.linalg.norm(values, axis=1)))
    if radius_median > 100.0:
        da = da / EARTH_RADIUS_KM
    elif not 0.8 < radius_median < 20.0:
        raise ValueError(
            f"Could not determine Arase position units; median radius={radius_median:g}."
        )
    return (
        _nearest_values(da, target_times, tolerance="120s"),
        source_files,
        source_version,
    )


def _load_t04_parameters(
    day: pd.Timestamp, target_times: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    from cdasws import CdasWs
    from cdasws.datarepresentation import DataRepresentation

    trange = [
        day.isoformat().replace("+00:00", "Z"),
        (day + pd.Timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    ]
    def cdas_dataset(dataset_id: str, variables: list[str]) -> xr.Dataset:
        client = CdasWs(timeout=60.0, disable_cache=True)
        status, result = client.get_data(
            dataset_id,
            variables,
            trange[0],
            trange[1],
            dataRepresentation=DataRepresentation.XARRAY,
        )
        code = status if isinstance(status, int) else status.get("http", {}).get(
            "status_code"
        )
        if code != 200 or not isinstance(result, xr.Dataset):
            raise RuntimeError(
                f"CDAWeb returned status={code!r} for T04 input {dataset_id}."
            )
        return result

    driver_path = t04_driver_daily_path(day)
    if driver_path.exists():
        with xr.open_dataset(driver_path) as cached:
            drivers = cached.load()
    else:
        omni = cdas_dataset(
            "OMNI_HRO2_1MIN",
            ["Pressure", "BY_GSM", "BZ_GSM"],
        )
        hourly = cdas_dataset("OMNI2_H0_MRG1HR", ["DST1800"])
        pressure_source = _as_data_array(omni["Pressure"], "Pressure")
        by_source = _as_data_array(omni["BY_GSM"], "BY_GSM")
        bz_source = _as_data_array(omni["BZ_GSM"], "BZ_GSM")
        dst_source = _as_data_array(hourly["DST1800"], "DST1800")
        drivers = xr.Dataset(
            {
                "Pressure": pressure_source.rename({"time": "omni_time"}),
                "BY_GSM": by_source.rename({"time": "omni_time"}),
                "BZ_GSM": bz_source.rename({"time": "omni_time"}),
                "DST1800": dst_source.rename({"time": "dst_time"}),
            },
            attrs={
                "OMNI_HRO2_1MIN_Data_version": str(
                    omni.attrs.get("Data_version", "")
                ),
                "OMNI2_H0_MRG1HR_Data_version": str(
                    hourly.attrs.get("Data_version", "")
                ),
                "source_datasets": "OMNI_HRO2_1MIN; OMNI2_H0_MRG1HR",
                "cache_date_utc": day.strftime("%Y-%m-%d"),
                "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            },
        )
        save_tracers_dataset(drivers, driver_path, overwrite=True)

    required = {"Pressure", "BY_GSM", "BZ_GSM", "DST1800"}
    missing = sorted(required - set(drivers.data_vars))
    if missing:
        raise KeyError(f"Cached T04 driver file lacks variables: {missing}.")
    pressure_source = drivers["Pressure"].rename({"omni_time": "time"})
    by_source = drivers["BY_GSM"].rename({"omni_time": "time"})
    bz_source = drivers["BZ_GSM"].rename({"omni_time": "time"})
    dst_source = drivers["DST1800"].rename({"dst_time": "time"})
    pressure = _nearest_values(
        pressure_source,
        target_times,
        tolerance="90s",
    )
    dst = _nearest_values(
        dst_source,
        target_times,
        tolerance="31min",
    )
    by = _nearest_values(
        by_source,
        target_times,
        tolerance="90s",
    )
    bz = _nearest_values(
        bz_source,
        target_times,
        tolerance="90s",
    )
    if pressure.ndim != 1 or dst.ndim != 1 or by.ndim != 1 or bz.ndim != 1:
        raise ValueError("T04 scalar input variables have unexpected dimensions.")

    w_year = _load_or_create_w_year(day.year)
    w_values = np.column_stack(
        [
            _nearest_values(w_year[name], target_times, tolerance="180s")
            for name in T04_COLUMNS[4:]
        ]
    )
    provenance = {
        "OMNI_HRO2_1MIN_Data_version": str(
            drivers.attrs.get("OMNI_HRO2_1MIN_Data_version", "")
        ),
        "OMNI2_H0_MRG1HR_Data_version": str(
            drivers.attrs.get("OMNI2_H0_MRG1HR_Data_version", "")
        ),
        "OMNI_Dst_cache": str(driver_path),
        "TS05_W_cache": str(ts05_w_yearly_path(day.year)),
    }
    return np.column_stack((pressure, dst, by, bz, w_values)), provenance


def _sm_coordinates(geopack: Any, xyz_gsm: Sequence[float]) -> np.ndarray:
    x, y, z = geopack.smgsm_vectorized(*xyz_gsm, j=-1)
    return np.asarray([x, y, z], dtype=float).reshape(3)


def _mlt_and_latitude(xyz_sm: Sequence[float]) -> tuple[float, float]:
    x, y, z = np.asarray(xyz_sm, dtype=float)
    radius_xy = np.hypot(x, y)
    mlt = (np.degrees(np.arctan2(y, x)) / 15.0 + 12.0) % 24.0
    latitude = np.degrees(np.arctan2(z, radius_xy))
    return float(mlt), float(latitude)


def compute_tracers_sm_track(tracers_orbit: xr.Dataset) -> xr.Dataset:
    """Compute TRACERS SM position, MLT, and signed magnetic latitude only."""
    required = {"X_GEO", "Y_GEO", "Z_GEO"}
    missing = sorted(required - set(tracers_orbit.data_vars))
    if missing:
        raise KeyError(f"TRACERS orbit lacks required variables: {missing}.")
    if "time" not in tracers_orbit.coords:
        raise KeyError("TRACERS orbit lacks a time coordinate.")
    times = np.asarray(tracers_orbit.time.values, dtype="datetime64[ns]")
    tracer_geo = np.column_stack(
        [
            np.asarray(tracers_orbit[name].values, dtype=float)
            for name in ("X_GEO", "Y_GEO", "Z_GEO")
        ]
    )
    n_time = times.size
    tracer_gsm = np.full((n_time, 3), np.nan, dtype=float)
    tracer_sm = np.full((n_time, 3), np.nan, dtype=float)
    mlt = np.full(n_time, np.nan, dtype=float)
    latitude = np.full(n_time, np.nan, dtype=float)
    hemisphere = np.zeros(n_time, dtype=np.int8)
    geopack = _load_local_geopack()
    for index, timestamp in enumerate(times):
        geo = tracer_geo[index]
        if not np.all(np.isfinite(geo)):
            continue
        geopack.recalc(pd.Timestamp(timestamp).timestamp())
        xgsm, ygsm, zgsm = geopack.geogsm_vectorized(*geo, j=1)
        tracer_gsm[index] = np.asarray([xgsm, ygsm, zgsm], dtype=float)
        tracer_sm[index] = _sm_coordinates(geopack, tracer_gsm[index])
        mlt[index], latitude[index] = _mlt_and_latitude(tracer_sm[index])
        hemisphere[index] = 1 if latitude[index] >= 0.0 else -1

    result = xr.Dataset(
        {
            "Position_GSM_RE": (("time", "component"), tracer_gsm),
            "Position_SM_RE": (("time", "component"), tracer_sm),
            "MLT_SM": ("time", mlt),
            "MagneticLatitude_SM": ("time", latitude),
            "Hemisphere_SM": ("time", hemisphere),
        },
        coords={
            "time": times,
            "component": np.asarray(["x", "y", "z"]),
        },
        attrs={
            "coordinate_source": "TRACERS SSCWeb GEO position",
            "coordinate_transform": "geopack-vectorize GEO->GSM->SM",
            "external_field_model": "none; spacecraft position transform only",
            "mlt_definition": "mod(atan2(y_SM, x_SM)/15 deg + 12 h, 24 h)",
            "latitude_definition": "geocentric magnetic latitude in SM",
        },
    )
    result["Position_GSM_RE"].attrs["units"] = "Re"
    result["Position_SM_RE"].attrs["units"] = "Re"
    result["MLT_SM"].attrs["units"] = "hour"
    result["MagneticLatitude_SM"].attrs["units"] = "deg"
    result["Hemisphere_SM"].attrs["values"] = "-1=south, 0=invalid, 1=north"
    return result


def compute_arase_t04_footprints(
    date: str | pd.Timestamp,
    tracers_orbit: xr.Dataset,
    *,
    probe: int,
    maxloop: int = 5000,
    trace_workers: int = 1,
    trace_chunk_size: int = 48,
) -> xr.Dataset:
    """Map Arase to the instantaneous TRACERS-radius sphere with T04+IGRF."""
    day = pd.Timestamp(date)
    if day.tzinfo is None:
        day = day.tz_localize("UTC")
    else:
        day = day.tz_convert("UTC")
    day = day.normalize()
    trace_workers = int(trace_workers)
    trace_chunk_size = int(trace_chunk_size)
    if not 1 <= trace_workers <= 16:
        raise ValueError("trace_workers must be between 1 and 16.")
    if trace_chunk_size <= 0:
        raise ValueError("trace_chunk_size must be positive.")
    required = {"X_GEO", "Y_GEO", "Z_GEO", "Altitude"}
    missing = sorted(required - set(tracers_orbit.data_vars))
    if missing:
        raise KeyError(f"TRACERS orbit lacks required variables: {missing}.")
    times = np.asarray(tracers_orbit.time.values, dtype="datetime64[ns]")
    if times.size == 0:
        raise ValueError("TRACERS orbit has no samples for the selected day.")
    trange = [
        day.isoformat().replace("+00:00", "Z"),
        (day + pd.Timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    ]

    # Import PySPEDAS first, then replace its legacy geopack namespace with the
    # workspace vectorized implementation used for tracing.
    arase_gsm, arase_source_files, arase_source_version = (
        _load_arase_position_gsm(trange, times)
    )
    parmod, t04_source_provenance = _load_t04_parameters(day, times)

    n_time = times.size
    nan_xyz = lambda: np.full((n_time, 3), np.nan, dtype=float)
    tracers_gsm = nan_xyz()
    tracers_sm = nan_xyz()
    arase_sm = nan_xyz()
    north_gsm, south_gsm = nan_xyz(), nan_xyz()
    north_sm, south_sm = nan_xyz(), nan_xyz()
    selected_gsm, selected_sm = nan_xyz(), nan_xyz()
    tracers_mlt = np.full(n_time, np.nan)
    tracers_lat = np.full(n_time, np.nan)
    footprint_mlt = np.full(n_time, np.nan)
    footprint_lat = np.full(n_time, np.nan)
    selected_hemisphere = np.zeros(n_time, dtype=np.int8)
    north_status = np.full(n_time, -1, dtype=np.int16)
    south_status = np.full(n_time, -1, dtype=np.int16)
    trace_input_valid = np.zeros(n_time, dtype=bool)

    tracer_geo = np.column_stack(
        [
            np.asarray(tracers_orbit[name].values, dtype=float)
            for name in ("X_GEO", "Y_GEO", "Z_GEO")
        ]
    )
    target_radius = np.linalg.norm(tracer_geo, axis=1)

    valid = (
        np.all(np.isfinite(tracer_geo), axis=1)
        & np.all(np.isfinite(arase_gsm), axis=1)
        & np.all(np.isfinite(parmod), axis=1)
        & np.isfinite(target_radius)
        & (target_radius > 1.0)
    )
    valid_indices = np.flatnonzero(valid)
    chunks = [
        valid_indices[start : start + trace_chunk_size]
        for start in range(0, valid_indices.size, trace_chunk_size)
    ]
    trace_workers_used = min(trace_workers, len(chunks)) if chunks else 0

    def assign_chunk(
        indices: np.ndarray, chunk_result: dict[str, np.ndarray]
    ) -> None:
        targets = {
            "tracers_gsm": tracers_gsm,
            "tracers_sm": tracers_sm,
            "arase_sm": arase_sm,
            "north_gsm": north_gsm,
            "south_gsm": south_gsm,
            "north_sm": north_sm,
            "south_sm": south_sm,
            "selected_gsm": selected_gsm,
            "selected_sm": selected_sm,
            "tracers_mlt": tracers_mlt,
            "tracers_lat": tracers_lat,
            "footprint_mlt": footprint_mlt,
            "footprint_lat": footprint_lat,
            "selected_hemisphere": selected_hemisphere,
            "north_status": north_status,
            "south_status": south_status,
            "trace_input_valid": trace_input_valid,
        }
        for name, target in targets.items():
            target[indices] = chunk_result[name]

    if trace_workers == 1 or len(chunks) <= 1:
        _initialize_t04_worker()
        for indices in chunks:
            result_indices, chunk_result = _trace_arase_chunk(
                indices,
                times[indices],
                tracer_geo[indices],
                arase_gsm[indices],
                parmod[indices],
                target_radius[indices],
                int(maxloop),
            )
            assign_chunk(result_indices, chunk_result)
    elif chunks:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(trace_workers, len(chunks)),
            mp_context=context,
            initializer=_initialize_t04_worker,
        ) as executor:
            futures = {
                executor.submit(
                    _trace_arase_chunk,
                    indices,
                    times[indices],
                    tracer_geo[indices],
                    arase_gsm[indices],
                    parmod[indices],
                    target_radius[indices],
                    int(maxloop),
                ): indices
                for indices in chunks
            }
            for future in as_completed(futures):
                result_indices, chunk_result = future.result()
                assign_chunk(result_indices, chunk_result)

    component = np.asarray(["x", "y", "z"])
    dataset = xr.Dataset(
        {
            "tracers_position_geo_re": (("time", "component"), tracer_geo),
            "tracers_position_gsm_re": (("time", "component"), tracers_gsm),
            "tracers_position_sm_re": (("time", "component"), tracers_sm),
            "tracers_mlt": ("time", tracers_mlt),
            "tracers_magnetic_latitude": ("time", tracers_lat),
            "tracers_altitude_km": (
                "time",
                np.asarray(tracers_orbit.Altitude.values, dtype=float),
            ),
            "conjugate_position_gsm_re": (("time", "component"), arase_gsm),
            "conjugate_position_sm_re": (("time", "component"), arase_sm),
            "conjugate_footprint_north_gsm_re": (
                ("time", "component"),
                north_gsm,
            ),
            "conjugate_footprint_south_gsm_re": (
                ("time", "component"),
                south_gsm,
            ),
            "conjugate_footprint_north_sm_re": (("time", "component"), north_sm),
            "conjugate_footprint_south_sm_re": (("time", "component"), south_sm),
            "conjugate_footprint_selected_gsm_re": (
                ("time", "component"),
                selected_gsm,
            ),
            "conjugate_footprint_selected_sm_re": (
                ("time", "component"),
                selected_sm,
            ),
            "conjugate_footprint_mlt": ("time", footprint_mlt),
            "conjugate_footprint_magnetic_latitude": ("time", footprint_lat),
            "selected_hemisphere": ("time", selected_hemisphere),
            "trace_status_north": ("time", north_status),
            "trace_status_south": ("time", south_status),
            "trace_input_valid": ("time", trace_input_valid),
            "t04_parmod": (("time", "parameter"), parmod),
        },
        coords={
            "time": times,
            "component": component,
            "parameter": np.asarray(T04_COLUMNS),
        },
        attrs={
            "conjugate_satellite": "Arase",
            "conjugate_source_version": arase_source_version,
            "conjugate_source_files": json.dumps(list(arase_source_files)),
            "t04_source_provenance": json.dumps(t04_source_provenance),
            "tracers_probe": int(probe),
            "external_field_model": "T04",
            "internal_field_model": "IGRF",
            "mapping_surface": "instantaneous spherical TRACERS geocentric radius",
            "earth_radius_km": EARTH_RADIUS_KM,
            "mlt_definition": "mod(atan2(y_SM, x_SM)/15 deg + 12 h, 24 h)",
            "latitude_definition": "geocentric magnetic latitude in SM",
            "interpolation": (
                "Arase nearest <=120 s; OMNI nearest <=90 s; "
                "Kyoto Dst nearest <=31 min; W1-W6 nearest <=180 s"
            ),
            "trace_status_codes": (
                "0=inner boundary; 1=outer boundary; 2=maxloop; "
                "-1=not attempted/no endpoint; -2=Arase inside target sphere"
            ),
            "trace_workers_requested": trace_workers,
            "trace_workers_used": trace_workers_used,
            "trace_chunk_size": trace_chunk_size,
        },
    )
    units = {
        "tracers_mlt": "hour",
        "conjugate_footprint_mlt": "hour",
        "tracers_magnetic_latitude": "deg",
        "conjugate_footprint_magnetic_latitude": "deg",
        "tracers_altitude_km": "km",
    }
    for name, unit in units.items():
        dataset[name].attrs["units"] = unit
    return dataset


def compute_conjugate_satellite_day(
    satellite: str,
    date: str | pd.Timestamp,
    tracers_orbit: xr.Dataset,
    *,
    probe: int,
    trace_workers: int = 1,
    trace_chunk_size: int = 48,
) -> xr.Dataset:
    """Dispatch one registry satellite to its daily footprint implementation."""
    satellite = normalize_conjugate_satellites([satellite])[0]
    if satellite == "Arase":
        return compute_arase_t04_footprints(
            date,
            tracers_orbit,
            probe=probe,
            trace_workers=trace_workers,
            trace_chunk_size=trace_chunk_size,
        )
    raise NotImplementedError(satellite)


__all__ = [
    "CONJUGATE_SATELLITE_REGISTRY",
    "ConjugateSatelliteSpec",
    "compute_arase_t04_footprints",
    "compute_conjugate_satellite_day",
    "compute_tracers_sm_track",
    "conjugacy_daily_path",
    "normalize_conjugate_satellites",
    "t04_driver_daily_path",
    "ts05_w_yearly_path",
]
