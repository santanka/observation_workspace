#!/usr/bin/env python3
"""Resumable batch execution of the Arase E/B analysis notebook.

The original notebook remains the authoritative single-range workflow.  This
runner parameterizes an in-memory copy, executes each range in an isolated
kernel process, records terminal status, and aggregates range/phase results.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "statistical_analysis_arase" / "Arase_BE_dispersion_relation_second.ipynb"
DEFAULT_RANGES = Path(
    "/mnt/j/statistical_analysis_arase/preanalysis/valid_time_ranges/"
    "Arase_valid_time_ranges_20220901_000000_to_20221001_000000.json"
)
KAW_ROOT = Path("/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation")
DEFAULT_STATE_ROOT = KAW_ROOT / "run_state"
DEFAULT_OUTPUT_ROOT = KAW_ROOT / "auto"
NOTEBOOK_OUTPUT_ROOT = "/mnt/j/statistical_analysis_arase/preanalysis/KAW_observation/E_B_ratio_Arase"
DEFAULT_SPEDAS_DIR = Path("/mnt/j/observation_data")

ANALYSIS_TIMEOUT_SEC = 24 * 3600
DOWNLOAD_TIMEOUT_SEC = 15 * 60
DOWNLOAD_RETRIES = 3
RETRY_DELAYS_SEC = (30, 120, 300)
DEFAULT_DOWNLOAD_WORKERS = 8
POSTPASS_RETRIES = 2
POSTPASS_RETRY_DELAYS_SEC = (60, 300)

# The notebook is the analysis implementation, while this runner guarantees
# that automated ranges are not processed with an older KAW-selection contract.
REQUIRED_NOTEBOOK_ANALYSIS_MARKERS = (
    "EB_SPIN_LOGRMSE_MAX_DEX = 0.75",
    "EB_RESIDUAL_PEAK_MAX_DEX = 0.70",
    "logrmse_E_spinband_kaw",
    "n_E_spinband_kaw",
    "_fit_e_spinband_rows(E, freq, coherency, wco_sig95, mask_fit)",
    "_additional_eb_selection_mask(data_win)",
    "_additional_eb_selection_mask(data_random)",
    "_additional_eb_selection_mask(data)",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_key(ranges_path: Path, ranges_hash: str) -> str:
    """Return a filesystem-safe identity for one range manifest revision."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", ranges_path.stem).strip("._-")
    if not stem:
        stem = "valid_time_ranges"
    return f"{stem}_{ranges_hash[:12]}"


def resolve_run_paths(args: argparse.Namespace, ranges_hash: str) -> str:
    """Namespace default state and output paths by range-file content."""
    key = dataset_key(Path(args.ranges), ranges_hash)
    if args.state_dir is None:
        args.state_dir = DEFAULT_STATE_ROOT / key
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT / key
    return key


def load_ranges(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    ranges = payload.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError(f"No ranges found in {path}")

    seen: set[int] = set()
    previous_end = None
    for item in ranges:
        for key in ("range_id", "start_time", "end_time"):
            if key not in item:
                raise KeyError(f"Range lacks {key}: {item}")
        range_id = int(item["range_id"])
        if range_id in seen:
            raise ValueError(f"Duplicate range_id={range_id}")
        seen.add(range_id)
        start = datetime.fromisoformat(str(item["start_time"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(item["end_time"]).replace("Z", "+00:00"))
        if start >= end:
            raise ValueError(f"Non-increasing range_id={range_id}")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"Overlapping or unsorted range_id={range_id}")
        previous_end = end
    return payload, ranges


def selected_ranges(ranges: list[dict[str, Any]], ids: list[int] | None) -> list[dict[str, Any]]:
    if not ids:
        return ranges
    wanted = set(ids)
    selected = [item for item in ranges if int(item["range_id"]) in wanted]
    missing = wanted - {int(item["range_id"]) for item in selected}
    if missing:
        raise ValueError(f"Unknown range IDs: {sorted(missing)}")
    return selected


def range_label(item: dict[str, Any]) -> str:
    start = datetime.fromisoformat(str(item["start_time"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(item["end_time"]).replace("Z", "+00:00"))
    return f"{start:%Y-%m-%d}/{start:%H%M}-{end:%H%M}"


def range_tag(item: dict[str, Any]) -> str:
    start = datetime.fromisoformat(str(item["start_time"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(item["end_time"]).replace("Z", "+00:00"))
    return f"{start:%Y%m%d_%H%M%S}_to_{end:%Y%m%d_%H%M%S}"


def expected_summary(item: dict[str, Any], output_root: Path) -> Path:
    return (
        output_root
        / range_label(item)
        / "EB_ratio_toroidal_eachtime"
        / f"kappa_phase_summary_toroidal_{range_tag(item)}.csv"
    )


def status_path(state_dir: Path, item: dict[str, Any]) -> Path:
    return state_dir / "ranges" / f"range_{int(item['range_id']):03d}.json"


def compatible_complete(
    status: dict[str, Any], notebook_hash: str, ranges_hash: str, runner_hash: str
) -> bool:
    return (
        status.get("status") == "complete"
        and status.get("notebook_sha256") == notebook_hash
        and status.get("ranges_sha256") == ranges_hash
        and status.get("runner_sha256") == runner_hash
        and Path(status.get("summary_path", "")).is_file()
    )


def terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_child(command: list[str], log_path: Path, timeout: int, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write(f"\n[{utc_now()}] command: {' '.join(command)}\n")
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"[{utc_now()}] timeout after {timeout} s\n")
            terminate_process_group(proc)
            return 124


def transform_notebook_source(
    source: str, start: str, end: str, output_root: Path
) -> str:
    """Apply batch-only settings without changing the source notebook."""
    import re

    source, _ = re.subn(
        r"time_range_input\s*=\s*\[[^\n]+\]",
        f"time_range_input = [{start!r}, {end!r}]",
        source,
        count=1,
    )
    source = source.replace("PLOT_WAVELET_MEDIAN_PSD = True", "PLOT_WAVELET_MEDIAN_PSD = False")
    source = source.replace(NOTEBOOK_OUTPUT_ROOT, str(output_root))

    # Network access is confined to the prefetch stage.
    replacements = {
        "get_support_data=True)": "get_support_data=True, no_update=True)",
        "datatype='def')": "datatype='def', no_update=True)",
        "datatype='3dflux', level='l2')": "datatype='3dflux', level='l2', no_update=True)",
        "ergpy.pwe_hfa(trange=time_range, level='l3')":
            "ergpy.pwe_hfa(trange=time_range, level='l3', no_update=True)",
        "ergpy.pwe_hfa(trange=time_range, level='l2')":
            "ergpy.pwe_hfa(trange=time_range, level='l2', no_update=True)",
        "psp.projects.omni.data(trange=trange, datatype='1min')":
            "psp.projects.omni.data(trange=trange, datatype='1min', no_update=True)",
        "psp.projects.omni.data(trange=omni_hourly_date_trange(trange), datatype='hourly', prefix='omni_hourly_')":
            "psp.projects.omni.data(trange=omni_hourly_date_trange(trange), datatype='hourly', prefix='omni_hourly_', no_update=True)",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    # ATT is prefetched as daily text files. Load it from the local cache in
    # each isolated kernel, then prevent erg_cotrans from calling att() again.
    source = source.replace(
        "def load_arase_position_dsi(time_range):\n"
        "    ergpy.orb(trange=time_range, level='l2', datatype='def', no_update=True)",
        "def load_arase_position_dsi(time_range):\n"
        "    psp.projects.erg.att(\n"
        "        trange=time_range, level='l2', no_update=True,\n"
        "    )\n"
        "    ergpy.orb(trange=time_range, level='l2', datatype='def', no_update=True)",
        1,
    )
    source = source.replace(
        "    pos_unit_gsm = pos_gsm / np.sqrt((pos_gsm * pos_gsm).sum(dim='v_dim'))\n\n"
        "    psp.store_data(",
        "    pos_unit_gsm = pos_gsm / np.sqrt((pos_gsm * pos_gsm).sum(dim='v_dim'))\n\n"
        "    # noload=True also suppresses the Sun-vector preparation inside dsi2j2000.\n"
        "    # Reproduce that preparation from the already cached ORB data first.\n"
        "    pos_gse = get_tplot_da('erg_orb_l2_pos_gse').interp(time=pos_unit_gsm.time)\n"
        "    sundir_gse = np.array([1.496e8, 0.0, 0.0]) - pos_gse.data\n"
        "    sundir_gse /= np.linalg.norm(sundir_gse, axis=1, keepdims=True)\n"
        "    psp.store_data(\n"
        "        'sundir_gse',\n"
        "        data={'x': pos_unit_gsm.time.values, 'y': sundir_gse},\n"
        "    )\n"
        "    psp.cotrans(\n"
        "        name_in='sundir_gse',\n"
        "        name_out='sundir_j2000',\n"
        "        coord_in='gse',\n"
        "        coord_out='j2000',\n"
        "    )\n\n"
        "    psp.store_data(",
        1,
    )
    source = source.replace(
        "        out_coord='dsi',\n"
        "    )\n"
        "    pos_unit_dsi = psp.get_data('Arase_pos_unit_dsi'",
        "        out_coord='dsi',\n"
        "        noload=True,\n"
        "    )\n"
        "    pos_unit_dsi = psp.get_data('Arase_pos_unit_dsi'",
        1,
    )

    # Permit HFA interpolation only when the native bracket is strictly below 5 min.
    old_function = """def interp_unique_time(da, time_base, method='linear'):
    return dedup_and_sort(da).interp(time=time_base, method=method)
"""
    new_function = """def interp_unique_time(da, time_base, method='linear', max_gap=None):
    source = dedup_and_sort(da)
    out = source.interp(time=time_base, method=method)
    if max_gap is None:
        return out
    if source.ndim != 1:
        raise ValueError('max_gap interpolation currently requires a 1-D DataArray')
    source_time = pd.DatetimeIndex(pd.to_datetime(source.time.values)).to_numpy(
        dtype='datetime64[ns]'
    ).astype(np.int64)
    target_time = pd.DatetimeIndex(pd.to_datetime(time_base.values)).to_numpy(
        dtype='datetime64[ns]'
    ).astype(np.int64)
    finite_source = np.isfinite(np.asarray(source.values, dtype=float))
    valid_time = source_time[finite_source]
    allowed = np.zeros(target_time.size, dtype=bool)
    if valid_time.size:
        right = np.searchsorted(valid_time, target_time, side='left')
        exact_ok = (right < valid_time.size)
        exact = np.zeros(target_time.size, dtype=bool)
        exact[exact_ok] = valid_time[right[exact_ok]] == target_time[exact_ok]
        bracket = (right > 0) & (right < valid_time.size)
        bracket_idx = np.flatnonzero(bracket)
        gap_ns = pd.to_timedelta(max_gap).value
        allowed[bracket_idx] = (
            valid_time[right[bracket_idx]] - valid_time[right[bracket_idx] - 1]
        ) < gap_ns
        allowed |= exact
    return out.where(xr.DataArray(allowed, dims='time', coords={'time': time_base.values}))
"""
    source = source.replace(old_function, new_function, 1)
    source = source.replace(
        "ne_hfa = interp_unique_time(ND_electron_HFA, time_base)",
        "ne_hfa = interp_unique_time(ND_electron_HFA, time_base, max_gap='5min')",
        1,
    )

    # Pandas 3 may retain microsecond resolution in DatetimeIndex.  Convert
    # explicitly to ns before comparing with Timedelta.value (always ns).
    source = source.replace(
        "elapsed_ns = times.asi8 - times.asi8[0]",
        "time_ns = times.to_numpy(dtype='datetime64[ns]').astype(np.int64)\n"
        "    elapsed_ns = time_ns - time_ns[0]",
    )
    source = source.replace(
        ").asi8[0]",
        ").to_numpy(dtype='datetime64[ns]').astype(np.int64)[0]",
    )
    source = source.replace(
        "(detected_times.asi8 - time_origin_ns) // block_ns",
        "(detected_times.to_numpy(dtype='datetime64[ns]').astype(np.int64) "
        "- time_origin_ns) // block_ns",
    )

    insufficient_marker = '''        print(f"Reason: {reason}")
        return {
'''
    insufficient_replacement = '''        print(f"Reason: {reason}")
        qc_base = (
            f"insufficient_data_{direction}_{phase_mode}_"
            f"{time_str_start}_to_{time_str_end}"
        )
        qc_figure_path = out_dir / (qc_base + ".png")
        qc_fig, qc_ax = plt.subplots(figsize=(8, 4))
        qc_ax.axis("off")
        qc_ax.text(
            0.5, 0.5,
            f"{phase_title}\\nInsufficient data\\n"
            f"{n_detected_times} fit-valid samples "
            f"({detected_duration_sec:.3f} s)\\n"
            f"{n_occupied_blocks} occupied {block_duration} blocks\\n"
            f"{reason}",
            ha="center", va="center", transform=qc_ax.transAxes,
        )
        qc_fig.tight_layout()
        qc_fig.savefig(qc_figure_path, dpi=200, bbox_inches="tight")
        plt.close(qc_fig)
        return {
'''
    source = source.replace(insufficient_marker, insufficient_replacement, 1)
    source = source.replace(
        '            "bootstrap_figure": None,\n',
        '            "bootstrap_figure": None,\n'
        '            "qc_figure": qc_figure_path,\n',
        1,
    )

    # Preserve raw bootstrap draws and binned spectra for downstream statistics.
    marker = '    main_png_path = out_dir / (kappa_base + ".png")\n'
    insertion = marker + '''
    bootstrap_samples_path = out_dir / (kappa_base + "_bootstrap_samples.npz")
    np.savez_compressed(
        bootstrap_samples_path,
        kappa_E=E_boot["samples"],
        kappa_B=B_boot["samples"],
    )
    kbin_statistics_path = out_dir / (kappa_base + "_kbin_statistics.npz")
    np.savez_compressed(
        kbin_statistics_path,
        k_edges=k_edges,
        k_centers=k_centers,
        log10_E_mean=log10_E_mean_k,
        E_count=E_count_k,
        log10_B_mean=log10_B_mean_k,
        B_count=B_count_k,
        log10_EB_ratio_mean=log10_EB_ratio_mean_k,
        EB_ratio_count=EB_ratio_count_k,
        log10_Spara_mean=log10_Spara_mean_k,
        Spara_count=Spara_count_k,
        log10_W_wave_mean=log10_W_wave_mean_k,
        W_wave_count=W_wave_count_k,
    )
'''
    source = source.replace(marker, insertion, 1)
    source = source.replace(
        '        "bootstrap_figure": bootstrap_png_path,\n',
        '        "bootstrap_figure": bootstrap_png_path,\n'
        '        "qc_figure": None,\n'
        '        "bootstrap_samples": bootstrap_samples_path,\n'
        '        "kbin_statistics": kbin_statistics_path,\n',
        1,
    )
    return source


CONTEXT_CELL = r'''
# Batch-only environmental and spacecraft-position summaries.
def _describe_numeric(values, prefix):
    a = np.asarray(values, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    out = {
        f"{prefix}_n_total": int(a.size),
        f"{prefix}_n_valid": int(finite.size),
        f"{prefix}_valid_fraction": finite.size / a.size if a.size else np.nan,
    }
    if finite.size:
        q16, median, q84 = np.percentile(finite, [16, 50, 84])
        out.update({
            f"{prefix}_mean": float(np.mean(finite)),
            f"{prefix}_std": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
            f"{prefix}_median": float(median),
            f"{prefix}_q16": float(q16),
            f"{prefix}_q84": float(q84),
        })
    else:
        out.update({f"{prefix}_{key}": np.nan for key in ("mean", "std", "median", "q16", "q84")})
    return out


def _describe_mlt(values, prefix):
    a = np.asarray(values, dtype=float).ravel()
    finite = a[np.isfinite(a)] % 24.0
    out = {
        f"{prefix}_n_total": int(a.size),
        f"{prefix}_n_valid": int(finite.size),
        f"{prefix}_valid_fraction": finite.size / a.size if a.size else np.nan,
    }
    if not finite.size:
        out.update({
            f"{prefix}_circular_mean_hour": np.nan,
            f"{prefix}_circular_std_hour": np.nan,
            f"{prefix}_mean_sin": np.nan,
            f"{prefix}_mean_cos": np.nan,
        })
        return out
    angle = finite * (2.0 * np.pi / 24.0)
    mean_sin = float(np.mean(np.sin(angle)))
    mean_cos = float(np.mean(np.cos(angle)))
    resultant = np.hypot(mean_sin, mean_cos)
    mean_hour = (np.arctan2(mean_sin, mean_cos) % (2.0 * np.pi)) * 24.0 / (2.0 * np.pi)
    if np.isclose(mean_hour, 24.0):
        mean_hour = 0.0
    circ_std = np.sqrt(-2.0 * np.log(max(resultant, np.finfo(float).tiny))) * 24.0 / (2.0 * np.pi)
    out.update({
        f"{prefix}_circular_mean_hour": float(mean_hour),
        f"{prefix}_circular_std_hour": float(circ_std),
        f"{prefix}_mean_sin": mean_sin,
        f"{prefix}_mean_cos": mean_cos,
    })
    return out


def _position_arrays_at(target_time):
    pos_gsm = psp.get_data('erg_orb_l2_pos_gsm', xarray=True).sortby('time')
    pos_rmlatmlt = psp.get_data('erg_orb_l2_pos_rmlatmlt', xarray=True).sortby('time')
    pos_l = psp.get_data('erg_orb_l2_pos_Lm', xarray=True).sortby('time')
    target = xr.DataArray(pd.to_datetime(target_time), dims='time', coords={'time': pd.to_datetime(target_time)})
    return (
        pos_gsm.interp(time=target),
        pos_rmlatmlt.interp(time=target),
        pos_l.interp(time=target),
    )


def _environment_arrays():
    arrays = {
        'B_total_nT': np.asarray(ds_par_window['B_total_nT'].values, dtype=float),
        'number_density_HFA_cc': np.asarray(ds_par_window['number_density_HFA_cc'].values, dtype=float),
        'ion_plasma_beta_HFA': np.asarray(ds_par_window['ion_plasma_beta_HFA'].values, dtype=float),
        'i_e_temp_ratio': np.asarray(ds_par_window['i-e_temp_ratio'].values, dtype=float),
        'proton_cycl_freq_Hz': np.asarray(ds_par_window['proton_cycl_freq_Hz'].values, dtype=float),
        'temp_ion_eV': np.asarray(ds_par_window['temp_ion_eV'].values, dtype=float),
        'temp_electron_eV': np.asarray(ds_par_window['temp_electron_eV'].values, dtype=float),
        'ion_mass_kg': np.asarray(ds_par_window['ion_mass_kg'].values, dtype=float),
        'Alfven_speed_HFA_ms': np.asarray(ds_vel_window['Alfven_speed_HFA'].values, dtype=float),
        'ion_thermal_speed_ms': np.asarray(ds_vel_window['ion_thermal_speed'].values, dtype=float),
        'electron_thermal_speed_ms': np.asarray(ds_vel_window['electron_thermal_speed'].values, dtype=float),
        'perp_sys_speed_ms': np.asarray(ds_vel_window['perp_sys_speed'].values, dtype=float),
        'perp_ion_speed_ms': np.asarray(ds_vel_window['perp_ion_speed'].values, dtype=float),
    }
    for name in ds_par_window.data_vars:
        if name.startswith('geomag_'):
            arrays[name] = np.asarray(ds_par_window[name].values, dtype=float)
    return arrays


def _add_position_summary(out, gsm, rmlatmlt, lm, prefix, selected=None):
    def take(a):
        values = np.asarray(a, dtype=float)
        return values if selected is None else values[selected]
    g = take(gsm.values)
    r = take(rmlatmlt.values)
    l = take(lm.values)
    for i, name in enumerate(('gsm_x_RE', 'gsm_y_RE', 'gsm_z_RE')):
        out.update(_describe_numeric(g[:, i], f'{prefix}_{name}'))
    out.update(_describe_numeric(r[:, 0], f'{prefix}_R_RE'))
    out.update(_describe_numeric(r[:, 1], f'{prefix}_MLAT_deg'))
    out.update(_describe_mlt(r[:, 2], f'{prefix}_MLT'))
    # CDF CATDESC specifies descending pitch angle: 90, 60, 30 deg.
    for i, pitch in enumerate((90, 60, 30)):
        out.update(_describe_numeric(l[:, i], f'{prefix}_L_{pitch}deg'))
    finite_g = np.isfinite(g).all(axis=1)
    if np.count_nonzero(finite_g) > 1:
        cov = np.cov(g[finite_g], rowvar=False, ddof=1)
        for i, left in enumerate(('x', 'y', 'z')):
            for j, right in enumerate(('x', 'y', 'z')):
                if j >= i:
                    out[f'{prefix}_gsm_cov_{left}{right}_RE2'] = float(cov[i, j])


context_time = pd.to_datetime(data['time'].values)
pos_gsm_aligned, pos_rmlatmlt_aligned, pos_l_aligned = _position_arrays_at(context_time)
pos_gsm_native = psp.get_data('erg_orb_l2_pos_gsm', xarray=True).sortby('time').sel(time=slice(t_min, t_max))
pos_rmlatmlt_native = psp.get_data('erg_orb_l2_pos_rmlatmlt', xarray=True).sortby('time').sel(time=slice(t_min, t_max))
pos_l_native = psp.get_data('erg_orb_l2_pos_Lm', xarray=True).sortby('time').sel(time=slice(t_min, t_max))
environment = _environment_arrays()

range_context = {}
for name, values in environment.items():
    range_context.update(_describe_numeric(values, f'range_{name}'))
_add_position_summary(
    range_context, pos_gsm_native, pos_rmlatmlt_native, pos_l_native, 'range'
)

# Native HFA coverage and values; exactly 5 min is not interpolable.
hfa_raw = psp.get_data('erg_pwe_hfa_l3_1min_ne_mgf', xarray=True).sortby('time')
hfa_flag = psp.get_data('erg_pwe_hfa_l3_1min_quality_flag', xarray=True).sortby('time')
hfa_raw, hfa_flag = xr.align(hfa_raw, hfa_flag, join='inner')
hfa_native = hfa_raw.sel(time=slice(t_min, t_max))
hfa_native_flag = hfa_flag.sel(time=slice(t_min, t_max))
hfa_values = np.asarray(hfa_native.values, dtype=float).ravel()
hfa_flags = np.asarray(hfa_native_flag.values, dtype=float).ravel()
hfa_valid = np.isfinite(hfa_values) & np.isfinite(hfa_flags) & (hfa_flags < 1)
range_context.update(_describe_numeric(hfa_values[hfa_valid], 'range_HFA_native_density_cc'))
range_context['range_HFA_native_n_total'] = int(hfa_values.size)
range_context['range_HFA_native_n_valid'] = int(np.count_nonzero(hfa_valid))
range_context['range_HFA_native_valid_fraction'] = (
    np.count_nonzero(hfa_valid) / hfa_values.size if hfa_values.size else np.nan
)
range_context['range_HFA_native_n_bad_quality'] = int(np.count_nonzero(~np.isfinite(hfa_flags) | (hfa_flags >= 1)))
range_context['range_HFA_native_n_nonfinite_density'] = int(np.count_nonzero(~np.isfinite(hfa_values)))
hfa_valid_times = pd.DatetimeIndex(pd.to_datetime(hfa_native.time.values[hfa_valid]))
if hfa_valid_times.size > 1:
    hfa_time_ns = hfa_valid_times.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    range_context['range_HFA_native_max_gap_sec'] = float(np.max(np.diff(hfa_time_ns)) / 1e9)
else:
    range_context['range_HFA_native_max_gap_sec'] = np.nan

context_rows = []
for phase_mode in phase_modes:
    row = {'phase_mode': phase_mode, **range_context}
    phase_condition = make_phase_mask(phase_all, phase_mode)
    phase_coherent = (coherency_all >= wco_sig95_all) & phase_condition
    e_phase = np.where(phase_coherent[tmask], E_all[tmask], np.nan)
    b_phase = np.where(phase_coherent[tmask], B_all[tmask], np.nan)
    common_fit = (
        kappa_mask & np.isfinite(e_phase) & (e_phase > 0)
        & np.isfinite(b_phase) & (b_phase > 0)
    )
    detected_mask = np.any(common_fit, axis=1)
    full_selected = np.zeros(data.sizes.get('time', 0), dtype=bool)
    full_selected[np.flatnonzero(np.asarray(tmask, dtype=bool))[detected_mask]] = True
    for name, values in environment.items():
        row.update(_describe_numeric(values[full_selected], f'fit_{name}'))
    _add_position_summary(
        row, pos_gsm_aligned, pos_rmlatmlt_aligned, pos_l_aligned,
        'fit', selected=full_selected,
    )
    context_rows.append(row)

context_df = pd.DataFrame(context_rows)
context_filename = f'environment_position_summary_{direction}_{time_str_start}_to_{time_str_end}.csv'
context_path = out_dir / context_filename
context_df.to_csv(context_path, index=False)
summary_df = summary_df.drop(columns=[c for c in context_df.columns if c != 'phase_mode' and c in summary_df], errors='ignore')
summary_df = summary_df.merge(context_df, on='phase_mode', how='left', validate='one_to_one')
summary_df.to_csv(summary_path, index=False)
print(f'Environmental and position summaries saved to {context_path}')
print(f'Augmented phase summary saved to {summary_path}')
'''


def execute_notebook_once(args: argparse.Namespace) -> int:
    import nbformat
    from nbclient import NotebookClient

    notebook_path = Path(args.notebook).resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    n_parameterized = 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            had_parameter = "time_range_input" in cell.source and "=" in cell.source
            cell.source = transform_notebook_source(
                cell.source, args.start, args.end, Path(args.output_root)
            )
            if had_parameter and f"time_range_input = [{args.start!r}, {args.end!r}]" in cell.source:
                n_parameterized += 1
            cell.outputs = []
            cell.execution_count = None
    if n_parameterized != 1:
        raise RuntimeError(f"Expected one time_range_input assignment, found {n_parameterized}")
    transformed_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    required_transform_markers = (
        "max_gap='5min'",
        "bootstrap_samples_path =",
        "PLOT_WAVELET_MEDIAN_PSD = False",
        "no_update=True",
        "psp.projects.erg.att(",
        "name_out='sundir_j2000'",
        "noload=True",
        *REQUIRED_NOTEBOOK_ANALYSIS_MARKERS,
    )
    missing = [marker for marker in required_transform_markers if marker not in transformed_source]
    if missing:
        raise RuntimeError(f"Notebook batch transformation incomplete: {missing}")
    notebook.cells.append(nbformat.v4.new_code_cell(CONTEXT_CELL))

    os.environ["SPEDAS_DATA_DIR"] = str(Path(args.spedas_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name=args.kernel_name,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()
    return 0


@dataclass(frozen=True)
class ProductSpec:
    cadence: str
    pad_before: timedelta = timedelta(0)
    pad_after: timedelta = timedelta(0)
    max_block_units: int = 1


@dataclass(frozen=True)
class DownloadUnit:
    product: str
    key: str
    start: datetime
    end_exclusive: datetime
    range_ids: tuple[int, ...]


@dataclass(frozen=True)
class DownloadBlock:
    product: str
    units: tuple[DownloadUnit, ...]
    start: datetime
    end_exclusive: datetime
    range_ids: tuple[int, ...]

    @property
    def key(self) -> str:
        first = self.units[0].key
        last = self.units[-1].key
        return first if first == last else f"{first}_to_{last}"

    @property
    def request_end(self) -> datetime:
        # OMNIは月/半年fileの代表1日だけを要求し、同じremote名の大量列挙を避ける。
        cadence = PRODUCT_SPECS[self.product].cadence
        if cadence in {"month", "half_year"}:
            return self.start + timedelta(days=1) - timedelta(seconds=1)
        # Loaderのtrange終端で次のfile unitを拾わないよう、右端を1秒戻す。
        return self.end_exclusive - timedelta(seconds=1)


PRODUCT_SPECS: dict[str, ProductSpec] = {
    "pwe_efd_l2_64": ProductSpec("day", max_block_units=7),
    "mgf_l2_64hz": ProductSpec("hour", max_block_units=24),
    "orb_l2_def": ProductSpec("day", max_block_units=7),
    "att_l2": ProductSpec(
        "day", pad_before=timedelta(seconds=60),
        pad_after=timedelta(seconds=60), max_block_units=7,
    ),
    "lepe_l2_3dflux": ProductSpec("day", max_block_units=7),
    "lepi_l2_3dflux": ProductSpec("day", max_block_units=7),
    "pwe_hfa_l3": ProductSpec("day", max_block_units=7),
    "pwe_hfa_l2": ProductSpec("day", max_block_units=7),
    "omni_1min": ProductSpec(
        "month", pad_before=timedelta(hours=3),
        pad_after=timedelta(hours=3), max_block_units=1,
    ),
    "omni_hourly": ProductSpec(
        "half_year", pad_before=timedelta(hours=3),
        pad_after=timedelta(hours=3), max_block_units=1,
    ),
}
PRODUCTS = tuple(PRODUCT_SPECS)


def _parse_time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _floor_unit(value: datetime, cadence: str) -> datetime:
    if cadence == "hour":
        return value.replace(minute=0, second=0, microsecond=0)
    if cadence == "day":
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    if cadence == "month":
        return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if cadence == "half_year":
        month = 1 if value.month <= 6 else 7
        return value.replace(month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unknown cadence: {cadence}")


def _next_unit(value: datetime, cadence: str) -> datetime:
    if cadence == "hour":
        return value + timedelta(hours=1)
    if cadence == "day":
        return value + timedelta(days=1)
    if cadence == "month":
        if value.month == 12:
            return value.replace(year=value.year + 1, month=1)
        return value.replace(month=value.month + 1)
    if cadence == "half_year":
        if value.month == 1:
            return value.replace(month=7)
        return value.replace(year=value.year + 1, month=1)
    raise ValueError(f"Unknown cadence: {cadence}")


def _unit_key(value: datetime, cadence: str) -> str:
    if cadence == "hour":
        return value.strftime("%Y%m%dT%H")
    if cadence == "day":
        return value.strftime("%Y%m%d")
    if cadence == "month":
        return value.strftime("%Y%m")
    if cadence == "half_year":
        return f"{value.year:04d}H{1 if value.month == 1 else 2}"
    raise ValueError(f"Unknown cadence: {cadence}")


def download_tranges(start_text: str, end_text: str) -> dict[str, list[str]]:
    """Return the legacy per-range request intervals used by each product."""
    start = _parse_time(start_text)
    end = _parse_time(end_text)
    exact = [start.isoformat(), end.isoformat()]
    geomag_start = start - timedelta(hours=3)
    geomag_end = end + timedelta(hours=3)
    geomag = [geomag_start.isoformat(), geomag_end.isoformat()]
    hourly = [
        geomag_start.strftime("%Y-%m-%d"),
        (geomag_end.replace(hour=0, minute=0, second=0, microsecond=0)
         + timedelta(days=1)).strftime("%Y-%m-%d"),
    ]
    return {"exact": exact, "omni_1min": geomag, "omni_hourly": hourly}


def build_download_units(
    ranges: list[dict[str, Any]], product: str
) -> list[DownloadUnit]:
    """Map all selected ranges to unique physical file units for one product."""
    if product not in PRODUCT_SPECS:
        raise ValueError(f"Unknown product: {product}")
    spec = PRODUCT_SPECS[product]
    unit_ranges: dict[datetime, set[int]] = {}

    for item in ranges:
        range_id = int(item["range_id"])
        start = _parse_time(item["start_time"]) - spec.pad_before
        end = _parse_time(item["end_time"]) + spec.pad_after
        if start >= end:
            raise ValueError(f"Non-increasing padded range_id={range_id}")

        cursor = _floor_unit(start, spec.cadence)
        final_unit = _floor_unit(end - timedelta(microseconds=1), spec.cadence)
        while cursor <= final_unit:
            unit_ranges.setdefault(cursor, set()).add(range_id)
            cursor = _next_unit(cursor, spec.cadence)

    return [
        DownloadUnit(
            product=product,
            key=_unit_key(start, spec.cadence),
            start=start,
            end_exclusive=_next_unit(start, spec.cadence),
            range_ids=tuple(sorted(unit_ranges[start])),
        )
        for start in sorted(unit_ranges)
    ]


def block_download_units(
    units: list[DownloadUnit], max_block_units: int | None = None
) -> list[DownloadBlock]:
    """Merge consecutive file units while retaining bounded retry granularity."""
    if not units:
        return []
    products = {unit.product for unit in units}
    if len(products) != 1:
        raise ValueError("A download block cannot mix products")
    product = units[0].product
    if max_block_units is None:
        max_block_units = PRODUCT_SPECS[product].max_block_units
    max_block_units = int(max_block_units)
    if max_block_units < 1:
        raise ValueError("max_block_units must be >= 1")

    blocks: list[DownloadBlock] = []
    current: list[DownloadUnit] = []

    def append_current() -> None:
        if not current:
            return
        range_ids = sorted({rid for unit in current for rid in unit.range_ids})
        blocks.append(DownloadBlock(
            product=product,
            units=tuple(current),
            start=current[0].start,
            end_exclusive=current[-1].end_exclusive,
            range_ids=tuple(range_ids),
        ))

    for unit in sorted(units, key=lambda value: value.start):
        is_consecutive = bool(current) and current[-1].end_exclusive == unit.start
        if current and (not is_consecutive or len(current) >= max_block_units):
            append_current()
            current = []
        current.append(unit)
    append_current()
    return blocks


def build_download_plan(ranges: list[dict[str, Any]]) -> list[DownloadBlock]:
    plan: list[DownloadBlock] = []
    for product in PRODUCTS:
        plan.extend(block_download_units(build_download_units(ranges, product)))
    return plan


def download_unit_marker(state_dir: Path, unit: DownloadUnit) -> Path:
    return state_dir / "downloads" / unit.product / f"{unit.key}.json"


def compatible_download_unit(marker: Path, unit: DownloadUnit) -> bool:
    if not marker.is_file():
        return False
    try:
        prior = json.loads(marker.read_text())
    except Exception:
        return False
    files = prior.get("files")
    return (
        prior.get("status") == "complete"
        and prior.get("product") == unit.product
        and prior.get("unit_key") == unit.key
        and prior.get("unit_start") == unit.start.isoformat()
        and prior.get("unit_end_exclusive") == unit.end_exclusive.isoformat()
        and isinstance(files, list)
        and bool(files)
        and all(Path(path).is_file() for path in files)
    )


def known_missing_download_unit(marker: Path, unit: DownloadUnit) -> bool:
    if not marker.is_file():
        return False
    try:
        prior = json.loads(marker.read_text())
    except Exception:
        return False
    return (
        prior.get("status") == "missing_remote"
        and prior.get("product") == unit.product
        and prior.get("unit_key") == unit.key
        and prior.get("unit_start") == unit.start.isoformat()
        and prior.get("unit_end_exclusive") == unit.end_exclusive.isoformat()
    )


def _normalize_download_files(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, (str, Path)):
        values = [result]
    elif hasattr(result, "tolist"):
        values = result.tolist()
    else:
        values = list(result)
    return [str(value) for value in values if str(value)]


def _files_for_unit(files: list[str], unit: DownloadUnit) -> list[str]:
    """Select files whose filename contains this physical unit's time token."""
    cadence = PRODUCT_SPECS[unit.product].cadence
    if cadence == "hour":
        token = unit.start.strftime("%Y%m%d%H")
    elif cadence == "day":
        token = unit.start.strftime("%Y%m%d")
    elif cadence in {"month", "half_year"}:
        token = unit.start.strftime("%Y%m01")
    else:
        raise ValueError(f"Unknown cadence: {cadence}")
    return [path for path in files if token in Path(path).name]


def _product_request_trange(product: str, start: str, end: str) -> list[str]:
    # pyspedas OMNI hourly delegates to yearlynames(), which only accepts dates.
    if product == "omni_hourly":
        return [
            _parse_time(start).strftime("%Y-%m-%d"),
            _parse_time(end).strftime("%Y-%m-%d"),
        ]
    return [start, end]


def _att_files_cover_trange(files: list[str], start: str, end: str) -> bool:
    """Return True when local ATT filenames cover every requested UTC day."""
    cursor = _floor_unit(_parse_time(start), "day")
    final = _floor_unit(_parse_time(end), "day")
    filenames = [Path(path).name for path in files]
    while cursor <= final:
        token = cursor.strftime("%Y%m%d")
        if not any(token in filename for filename in filenames):
            return False
        cursor = _next_unit(cursor, "day")
    return True


def prefetch_one(args: argparse.Namespace) -> int:
    os.environ["SPEDAS_DATA_DIR"] = str(Path(args.spedas_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import ergpyspedas.erg as ergpy
    import pyspedas as psp

    product = args.product
    trange = _product_request_trange(product, args.start, args.end)

    def prefetch_att() -> list[str]:
        local_files = _normalize_download_files(psp.projects.erg.att(
            trange=trange, level="l2", downloadonly=True, no_update=True,
        ))
        if _att_files_cover_trange(local_files, args.start, args.end):
            return local_files
        downloaded = _normalize_download_files(psp.projects.erg.att(
            trange=trange, level="l2", downloadonly=True,
        ))
        return list(dict.fromkeys(local_files + downloaded))

    calls = {
        "pwe_efd_l2_64": lambda: ergpy.pwe_efd(
            trange=trange, level="l2", datatype="64", coord="dsi",
            get_support_data=True, downloadonly=True,
        ),
        "mgf_l2_64hz": lambda: ergpy.mgf(
            trange=trange, level="l2", datatype="64hz", coord="dsi",
            get_support_data=True, downloadonly=True,
        ),
        "orb_l2_def": lambda: ergpy.orb(
            trange=trange, level="l2", datatype="def", downloadonly=True,
        ),
        "att_l2": prefetch_att,
        "lepe_l2_3dflux": lambda: ergpy.lepe(
            trange=trange, datatype="3dflux", level="l2", downloadonly=True,
        ),
        "lepi_l2_3dflux": lambda: ergpy.lepi(
            trange=trange, datatype="3dflux", level="l2", downloadonly=True,
        ),
        "pwe_hfa_l3": lambda: ergpy.pwe_hfa(
            trange=trange, level="l3", downloadonly=True,
        ),
        "pwe_hfa_l2": lambda: ergpy.pwe_hfa(
            trange=trange, level="l2", downloadonly=True,
        ),
        "omni_1min": lambda: psp.projects.omni.data(
            trange=trange, datatype="1min", downloadonly=True,
        ),
        "omni_hourly": lambda: psp.projects.omni.data(
            trange=trange, datatype="hourly", downloadonly=True,
        ),
    }
    if product not in calls:
        raise ValueError(f"Unknown product: {product}")
    files = _normalize_download_files(calls[product]())
    if not files:
        raise RuntimeError(
            f"No downloaded/local files returned for {product} "
            f"{args.start} -- {args.end}"
        )
    missing = [path for path in files if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"Downloader returned missing local files: {missing}")
    result = {
        "status": "complete", "product": product,
        "start_time": args.start, "end_time": args.end,
        "files": files, "finished_at": utc_now(),
    }
    if args.result_json:
        atomic_json(Path(args.result_json), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _write_unit_status(
    state_dir: Path,
    unit: DownloadUnit,
    status: str,
    block: DownloadBlock,
    **extra: Any,
) -> None:
    atomic_json(download_unit_marker(state_dir, unit), {
        "status": status,
        "product": unit.product,
        "unit_key": unit.key,
        "unit_start": unit.start.isoformat(),
        "unit_end_exclusive": unit.end_exclusive.isoformat(),
        "range_ids": list(unit.range_ids),
        "block_key": block.key,
        **extra,
    })


def _run_prefetch_block(
    args: argparse.Namespace,
    block: DownloadBlock,
    state_dir: Path,
    env: dict[str, str],
) -> tuple[DownloadBlock, tuple[DownloadUnit, ...]]:
    block_dir = state_dir / "downloads" / "_blocks" / block.product
    log_path = state_dir / "logs" / "downloads" / block.product / f"{block.key}.log"

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        started_at = utc_now()
        for unit in block.units:
            _write_unit_status(
                state_dir, unit, "running", block,
                attempt=attempt, started_at=started_at,
            )

        result_path = block_dir / f"{block.key}_attempt_{attempt}.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "_prefetch-one",
            "--start", block.start.isoformat(),
            "--end", block.request_end.isoformat(),
            "--product", block.product,
            "--spedas-dir", str(args.spedas_dir),
            "--result-json", str(result_path),
        ]
        code = run_child(command, log_path, args.download_timeout, env)
        files: list[str] = []
        unit_files: dict[str, list[str]] = {}
        if code == 0 and result_path.is_file():
            try:
                result = json.loads(result_path.read_text())
                files = [str(path) for path in result.get("files", [])]
                if not files or not all(Path(path).is_file() for path in files):
                    code = 1
                else:
                    unit_files = {
                        unit.key: _files_for_unit(files, unit) for unit in block.units
                    }
            except Exception:
                code = 1

        finished_at = utc_now()
        if code == 0:
            missing_units = tuple(
                unit for unit in block.units if not unit_files[unit.key]
            )
            for unit in block.units:
                matched = unit_files[unit.key]
                if matched:
                    _write_unit_status(
                        state_dir, unit, "complete", block,
                        attempt=attempt, started_at=started_at,
                        finished_at=finished_at, files=matched,
                    )
                else:
                    _write_unit_status(
                        state_dir, unit, "missing_remote", block,
                        attempt=attempt, started_at=started_at,
                        finished_at=finished_at, files=[],
                        reason="No CDF returned for this physical file unit",
                    )
            return block, missing_units

        status = "timeout" if code == 124 else "failed"
        for unit in block.units:
            _write_unit_status(
                state_dir, unit, status, block,
                attempt=attempt, started_at=started_at,
                finished_at=finished_at, returncode=code, files=files,
            )
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(RETRY_DELAYS_SEC[attempt - 1])

    return block, block.units


def run_prefetch(args: argparse.Namespace, ranges: list[dict[str, Any]]) -> set[int]:
    env = os.environ.copy()
    env["SPEDAS_DATA_DIR"] = str(args.spedas_dir)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    state_dir = Path(args.state_dir)

    total_units = 0
    completed_units = 0
    known_missing_units: list[DownloadUnit] = []
    pending_blocks: list[DownloadBlock] = []
    for product in PRODUCTS:
        units = build_download_units(ranges, product)
        total_units += len(units)
        pending_units = []
        for unit in units:
            marker = download_unit_marker(state_dir, unit)
            if not args.force and compatible_download_unit(marker, unit):
                completed_units += 1
            elif not args.force and known_missing_download_unit(marker, unit):
                known_missing_units.append(unit)
            else:
                pending_units.append(unit)
        pending_blocks.extend(block_download_units(pending_units))

    print(f"prefetch ranges          : {len(ranges)}")
    print(f"prefetch unique units    : {total_units}")
    print(f"prefetch cached units    : {completed_units}")
    print(f"prefetch known missing   : {len(known_missing_units)}")
    print(f"prefetch pending blocks  : {len(pending_blocks)}")

    failed_ranges = {
        range_id
        for unit in known_missing_units
        for range_id in unit.range_ids
    }
    if not pending_blocks:
        return failed_ranges

    worker_count = max(1, min(int(args.download_workers), len(pending_blocks)))
    print(f"prefetch download workers: {worker_count}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_prefetch_block, args, block, state_dir, env): block
            for block in pending_blocks
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                _, failed_units = future.result()
            except Exception as exc:
                failed_units = block.units
                traceback.print_exc()
                for unit in block.units:
                    _write_unit_status(
                        state_dir, unit, "failed", block,
                        finished_at=utc_now(), error=repr(exc), files=[],
                    )
            if not failed_units:
                print(
                    f"prefetch complete: {block.product} {block.key} "
                    f"({len(block.units)} units)"
                )
            else:
                failed_ranges.update(
                    range_id
                    for unit in failed_units
                    for range_id in unit.range_ids
                )
                print(
                    f"Prefetch incomplete: {block.product} {block.key}; "
                    f"missing/failed units: {[unit.key for unit in failed_units]}",
                    file=sys.stderr,
                )
    return failed_ranges


ATT_NETWORK_SIGNATURES = (
    "Connection error getting remote index",
    "ReadTimeoutError",
    "ConnectTimeoutError",
    "ConnectionError",
    "RemoteDisconnected",
    "NameResolutionError",
    "Temporary failure in name resolution",
    "SSLError",
)


def classify_range_failure(log_text: str, returncode: int) -> tuple[str, bool]:
    """Classify a failed range without treating scientific data gaps as retryable."""
    att_context = "/erg/att/txt/" in log_text or "erg_interpolate_att" in log_text
    att_network_error = att_context and any(
        signature in log_text for signature in ATT_NETWORK_SIGNATURES
    )
    if att_network_error:
        return "att_download_transient", True
    if returncode == 124:
        return "analysis_timeout", False
    if att_context and "No objects to concatenate" in log_text:
        return "att_data_unavailable", False
    if "No overlapping E64/B64 DSI segments" in log_text:
        return "no_overlapping_efd_mgf_waveform", False
    if "No EB64 segment survived DSI to FAC rotation" in log_text:
        return "no_valid_fac_segment", False
    if "zero-size array" in log_text:
        return "empty_input_data", False
    return "analysis_error", False


def _log_text_since(path: Path, offset: int) -> str:
    if not path.is_file():
        return ""
    return path.read_bytes()[offset:].decode("utf-8", errors="replace")


def retryable_analysis_ranges(
    state_dir: Path, ranges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected = []
    for item in ranges:
        marker = status_path(state_dir, item)
        if not marker.is_file():
            continue
        try:
            status = json.loads(marker.read_text())
        except Exception:
            continue
        if status.get("status") in {"failed", "timeout"} and status.get("retryable") is True:
            selected.append(item)
    return selected


def retryable_download_range_ids(
    state_dir: Path, ranges: list[dict[str, Any]]
) -> set[int]:
    """Return ranges backed by failed/timeout units; missing_remote is terminal."""
    selected_ids = {int(item["range_id"]) for item in ranges}
    retryable_ids: set[int] = set()
    for product in PRODUCTS:
        for unit in build_download_units(ranges, product):
            marker = download_unit_marker(state_dir, unit)
            if not marker.is_file():
                continue
            try:
                status = json.loads(marker.read_text()).get("status")
            except Exception:
                continue
            if status in {"failed", "timeout"}:
                retryable_ids.update(selected_ids.intersection(unit.range_ids))
    return retryable_ids


def run_ranges(args: argparse.Namespace, ranges: list[dict[str, Any]], ranges_hash: str) -> None:
    notebook = Path(args.notebook).resolve()
    notebook_hash = file_sha256(notebook)
    runner_hash = file_sha256(Path(__file__).resolve())
    state_dir = Path(args.state_dir)
    env = os.environ.copy()
    env["SPEDAS_DATA_DIR"] = str(args.spedas_dir)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    for item in ranges:
        marker = status_path(state_dir, item)
        prior_status: dict[str, Any] = {}
        if marker.exists():
            try:
                prior_status = json.loads(marker.read_text())
                if not args.force and compatible_complete(
                    prior_status, notebook_hash, ranges_hash, runner_hash
                ):
                    print(f"skip complete range {item['range_id']}")
                    continue
            except Exception:
                prior_status = {}

        summary = expected_summary(item, Path(args.output_root))
        started = time.monotonic()
        attempt = int(prior_status.get("attempt", 0)) + 1
        base_status = {
            "range_id": int(item["range_id"]),
            "start_time": item["start_time"],
            "end_time": item["end_time"],
            "notebook_sha256": notebook_hash,
            "ranges_sha256": ranges_hash,
            "runner_sha256": runner_hash,
            "summary_path": str(summary),
            "attempt": attempt,
            "started_at": utc_now(),
        }
        atomic_json(marker, {**base_status, "status": "running"})
        command = [
            sys.executable, str(Path(__file__).resolve()), "_execute-one",
            "--notebook", str(notebook),
            "--start", str(item["start_time"]), "--end", str(item["end_time"]),
            "--output-root", str(args.output_root),
            "--spedas-dir", str(args.spedas_dir), "--kernel-name", args.kernel_name,
        ]
        log_path = state_dir / "logs" / "ranges" / f"range_{int(item['range_id']):03d}.log"
        log_offset = log_path.stat().st_size if log_path.is_file() else 0
        code = run_child(command, log_path, args.analysis_timeout, env)
        elapsed = time.monotonic() - started
        terminal = {
            **base_status, "returncode": code,
            "elapsed_sec": elapsed, "finished_at": utc_now(),
        }
        if code == 0 and summary.is_file():
            state = "complete"
            terminal.update(status=state, retryable=False)
        else:
            state = "timeout" if code == 124 else "failed"
            failure_class, retryable = classify_range_failure(
                _log_text_since(log_path, log_offset), code
            )
            terminal.update(
                status=state, failure_class=failure_class, retryable=retryable
            )
        atomic_json(marker, terminal)
        suffix = (
            f", {terminal['failure_class']}, retryable={terminal['retryable']}"
            if state != "complete" else ""
        )
        print(
            f"range {item['range_id']}: {state} "
            f"(attempt {attempt}, {elapsed / 60:.1f} min{suffix})"
        )



def run_postpass_retries(
    args: argparse.Namespace, ranges: list[dict[str, Any]], ranges_hash: str
) -> None:
    """Retry transient download units and classified transient range failures."""
    max_rounds = max(0, int(args.postpass_retries))
    by_id = {int(item["range_id"]): item for item in ranges}

    for retry_round in range(1, max_rounds + 1):
        analysis_items = retryable_analysis_ranges(Path(args.state_dir), ranges)
        analysis_ids = {int(item["range_id"]) for item in analysis_items}
        download_ids = retryable_download_range_ids(Path(args.state_dir), ranges)
        candidate_ids = analysis_ids | download_ids
        if not candidate_ids:
            return

        delay_index = min(retry_round - 1, len(POSTPASS_RETRY_DELAYS_SEC) - 1)
        delay = POSTPASS_RETRY_DELAYS_SEC[delay_index]
        print(
            f"postpass retry round {retry_round}/{max_rounds}: "
            f"download_ranges={sorted(download_ids)}, "
            f"analysis_ranges={sorted(analysis_ids)}, wait={delay}s"
        )
        time.sleep(delay)

        if download_ids:
            download_items = [by_id[range_id] for range_id in sorted(download_ids)]
            retry_args = copy.copy(args)
            retry_args.force = False
            still_failed = run_prefetch(retry_args, download_items)
            ready_ids = download_ids - still_failed
            if ready_ids:
                run_ranges(
                    args, [by_id[range_id] for range_id in sorted(ready_ids)], ranges_hash
                )

        # Do not run a range whose required download unit is still unresolved.
        direct_analysis_ids = analysis_ids - download_ids
        if direct_analysis_ids:
            run_ranges(
                args,
                [by_id[range_id] for range_id in sorted(direct_analysis_ids)],
                ranges_hash,
            )


def aggregate(args: argparse.Namespace, ranges: list[dict[str, Any]]) -> None:
    import pandas as pd

    frames = []
    status_rows = []
    state_dir = Path(args.state_dir)
    for item in ranges:
        marker = status_path(state_dir, item)
        status = json.loads(marker.read_text()) if marker.exists() else {
            "range_id": int(item["range_id"]), "status": "pending"
        }
        status_rows.append(status)
        # Output directories may contain summaries produced by older/manual
        # notebook runs.  Only results accepted by this runner belong in the
        # batch master table.
        if status.get("status") != "complete":
            continue
        summary = expected_summary(item, Path(args.output_root))
        if not summary.is_file():
            continue
        values = pd.read_csv(summary).reset_index(drop=True)
        metadata = pd.DataFrame({
            "range_id": int(item["range_id"]),
            "range_start": item["start_time"],
            "range_end": item["end_time"],
            "range_duration_sec": float(item.get("duration_sec", float("nan"))),
            "source_summary": str(summary),
            "notebook_sha256": status.get("notebook_sha256"),
            "ranges_sha256": status.get("ranges_sha256"),
            "runner_sha256": status.get("runner_sha256"),
        }, index=values.index)
        frame = pd.concat([metadata, values], axis=1)
        frames.append(frame)

    aggregate_dir = state_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame(status_rows).sort_values("range_id")
    status_df.to_csv(aggregate_dir / "range_status.csv", index=False)
    try:
        status_df.to_parquet(aggregate_dir / "range_status.parquet", index=False)
    except ImportError:
        print("Parquet engine is unavailable; wrote CSV only")
    if frames:
        master = pd.concat(frames, ignore_index=True)
        master.to_csv(aggregate_dir / "arase_phase_master.csv", index=False)
        try:
            master.to_parquet(aggregate_dir / "arase_phase_master.parquet", index=False)
        except ImportError:
            print("Parquet engine is unavailable; wrote CSV only")
        print(f"master rows: {len(master)}")
    print(status_df["status"].value_counts(dropna=False).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ranges", type=Path, default=DEFAULT_RANGES)
        p.add_argument("--range-id", type=int, action="append")
        p.add_argument(
            "--state-dir", type=Path, default=None,
            help="Default: KAW_observation/run_state/<range-manifest key>",
        )
        p.add_argument(
            "--output-root", type=Path, default=None,
            help="Default: KAW_observation/auto/<range-manifest key>",
        )
        p.add_argument("--spedas-dir", type=Path, default=DEFAULT_SPEDAS_DIR)
        p.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="Prefetch, execute ranges, and aggregate")
    common(run)
    run.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    run.add_argument("--kernel-name", default="python3")
    run.add_argument("--skip-prefetch", action="store_true")
    run.add_argument("--download-timeout", type=int, default=DOWNLOAD_TIMEOUT_SEC)
    run.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)
    run.add_argument("--analysis-timeout", type=int, default=ANALYSIS_TIMEOUT_SEC)
    run.add_argument(
        "--postpass-retries", type=int, default=POSTPASS_RETRIES,
        help="Retry transient download/ATT failures after the first full pass",
    )

    prefetch = sub.add_parser("prefetch", help="Download required product file units")
    common(prefetch)
    prefetch.add_argument("--download-timeout", type=int, default=DOWNLOAD_TIMEOUT_SEC)
    prefetch.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS)

    agg = sub.add_parser("aggregate", help="Rebuild master tables")
    common(agg)

    one = sub.add_parser("_execute-one")
    one.add_argument("--notebook", required=True)
    one.add_argument("--start", required=True)
    one.add_argument("--end", required=True)
    one.add_argument("--output-root", required=True)
    one.add_argument("--spedas-dir", required=True)
    one.add_argument("--kernel-name", default="python3")

    download = sub.add_parser("_prefetch-one")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--product", required=True)
    download.add_argument("--spedas-dir", required=True)
    download.add_argument("--result-json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "_execute-one":
            return execute_notebook_once(args)
        if args.command == "_prefetch-one":
            return prefetch_one(args)

        payload, all_ranges = load_ranges(Path(args.ranges))
        ranges = selected_ranges(all_ranges, args.range_id)
        ranges_hash = file_sha256(Path(args.ranges))
        key = resolve_run_paths(args, ranges_hash)
        print(f"dataset: {key}")
        print(f"state_dir: {args.state_dir}")
        print(f"output_root: {args.output_root}")
        if args.command == "prefetch":
            return 1 if run_prefetch(args, ranges) else 0
        if args.command == "aggregate":
            aggregate(args, ranges)
            return 0
        runnable_ranges = ranges
        if not args.skip_prefetch:
            failed_ranges = run_prefetch(args, ranges)
            if failed_ranges:
                runnable_ranges = []
                for item in ranges:
                    range_id = int(item["range_id"])
                    if range_id not in failed_ranges:
                        runnable_ranges.append(item)
                        continue
                    atomic_json(status_path(Path(args.state_dir), item), {
                        "range_id": int(item["range_id"]),
                        "start_time": item["start_time"],
                        "end_time": item["end_time"],
                        "status": "failed_download",
                        "reason": "One or more required products failed for this range",
                        "finished_at": utc_now(),
                    })
        run_ranges(args, runnable_ranges, ranges_hash)
        run_postpass_retries(args, ranges, ranges_hash)
        # A partial run must not replace the master tables with only that subset.
        # Rebuild them from every range whose output is currently available.
        aggregate(args, all_ranges)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
