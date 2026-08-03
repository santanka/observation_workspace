"""Lightweight process worker for TRACERS T04 tracing.

This module intentionally avoids xarray, CDAWeb, PySPEDAS, and SpacePy imports
so many spawned trace workers can start concurrently without shared config I/O.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


_WORKER_GEOPACK: Any | None = None


def _load_local_geopack() -> Any:
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


def _sm_coordinates(geopack: Any, xyz_gsm: Sequence[float]) -> np.ndarray:
    x, y, z = geopack.smgsm_vectorized(*xyz_gsm, j=-1)
    return np.asarray([x, y, z], dtype=float).reshape(3)


def _mlt_and_latitude(xyz_sm: Sequence[float]) -> tuple[float, float]:
    x, y, z = np.asarray(xyz_sm, dtype=float)
    mlt = (np.degrees(np.arctan2(y, x)) / 15.0 + 12.0) % 24.0
    latitude = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return float(mlt), float(latitude)


def initialize_t04_worker() -> None:
    """Load an independent geopack global state in one trace worker."""
    global _WORKER_GEOPACK
    _WORKER_GEOPACK = _load_local_geopack()


def trace_arase_chunk(
    indices: np.ndarray,
    times: np.ndarray,
    tracer_geo: np.ndarray,
    arase_gsm: np.ndarray,
    parmod: np.ndarray,
    target_radius: np.ndarray,
    maxloop: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Trace one time chunk with the process-local geopack state."""
    global _WORKER_GEOPACK
    if _WORKER_GEOPACK is None:
        initialize_t04_worker()
    geopack = _WORKER_GEOPACK
    indices = np.asarray(indices, dtype=np.int64)
    n_chunk = indices.size
    nan_xyz = lambda: np.full((n_chunk, 3), np.nan, dtype=float)
    result: dict[str, np.ndarray] = {
        "tracers_gsm": nan_xyz(),
        "tracers_sm": nan_xyz(),
        "arase_sm": nan_xyz(),
        "north_gsm": nan_xyz(),
        "south_gsm": nan_xyz(),
        "north_sm": nan_xyz(),
        "south_sm": nan_xyz(),
        "selected_gsm": nan_xyz(),
        "selected_sm": nan_xyz(),
        "tracers_mlt": np.full(n_chunk, np.nan),
        "tracers_lat": np.full(n_chunk, np.nan),
        "footprint_mlt": np.full(n_chunk, np.nan),
        "footprint_lat": np.full(n_chunk, np.nan),
        "selected_hemisphere": np.zeros(n_chunk, dtype=np.int8),
        "north_status": np.full(n_chunk, -1, dtype=np.int16),
        "south_status": np.full(n_chunk, -1, dtype=np.int16),
        "trace_input_valid": np.zeros(n_chunk, dtype=bool),
    }

    for local_index in range(n_chunk):
        timestamp = times[local_index]
        geo = tracer_geo[local_index]
        ara = arase_gsm[local_index]
        parameters = parmod[local_index]
        r0 = float(target_radius[local_index])
        if (
            not np.all(np.isfinite(geo))
            or not np.all(np.isfinite(ara))
            or not np.all(np.isfinite(parameters))
            or not np.isfinite(r0)
            or r0 <= 1.0
        ):
            continue

        geopack.recalc(pd.Timestamp(timestamp).timestamp())
        xgsm, ygsm, zgsm = geopack.geogsm_vectorized(*geo, j=1)
        result["tracers_gsm"][local_index] = np.asarray(
            [xgsm, ygsm, zgsm], dtype=float
        )
        result["tracers_sm"][local_index] = _sm_coordinates(
            geopack, result["tracers_gsm"][local_index]
        )
        result["arase_sm"][local_index] = _sm_coordinates(geopack, ara)
        (
            result["tracers_mlt"][local_index],
            result["tracers_lat"][local_index],
        ) = _mlt_and_latitude(result["tracers_sm"][local_index])
        hemisphere = 1 if result["tracers_lat"][local_index] >= 0 else -1
        result["selected_hemisphere"][local_index] = hemisphere

        if np.linalg.norm(ara) <= r0 + 1.0e-4:
            result["north_status"][local_index] = -2
            result["south_status"][local_index] = -2
            continue
        result["trace_input_valid"][local_index] = True

        endpoints: list[tuple[np.ndarray, np.ndarray, int]] = []
        for direction in (+1, -1):
            try:
                xf, yf, zf, status = geopack.trace_vectorized(
                    *ara,
                    dir=direction,
                    rlim=30.0,
                    r0=r0,
                    parmod=parameters,
                    exname="t04",
                    inname="igrf",
                    maxloop=int(maxloop),
                    return_full_path=False,
                    strict_scalar_models=False,
                )
                endpoint_gsm = np.asarray([xf, yf, zf], dtype=float)
                endpoint_sm = _sm_coordinates(geopack, endpoint_gsm)
                endpoints.append((endpoint_gsm, endpoint_sm, int(status)))
            except Exception:
                continue

        for endpoint_gsm, endpoint_sm, status in endpoints:
            branch = "north" if endpoint_sm[2] >= 0 else "south"
            result[f"{branch}_status"][local_index] = status
            if status == 0:
                result[f"{branch}_gsm"][local_index] = endpoint_gsm
                result[f"{branch}_sm"][local_index] = endpoint_sm

        selected_branch = "north" if hemisphere > 0 else "south"
        result["selected_gsm"][local_index] = result[
            f"{selected_branch}_gsm"
        ][local_index]
        result["selected_sm"][local_index] = result[
            f"{selected_branch}_sm"
        ][local_index]
        if np.all(np.isfinite(result["selected_sm"][local_index])):
            (
                result["footprint_mlt"][local_index],
                result["footprint_lat"][local_index],
            ) = _mlt_and_latitude(result["selected_sm"][local_index])

    return indices, result


__all__ = ["initialize_t04_worker", "trace_arase_chunk"]
