# erg_mgf_spintone_rm.py
#
# Spin-tone removal for Arase/ERG MGF 64 Hz data.
#
# Core API:
#   remove_spintone_3comp(time, Bx, By, Bz, phase_rad, ...)
#
# Optional xarray wrapper:
#   remove_spintone_3comp_xr(ds, bx_name, by_name, bz_name, spin_phase_name, ...)
#
# time      : numpy array of datetime64 (1D)
# Bx,By,Bz  : numpy arrays of magnetic field components (1D, same length as time)
# phase_rad : numpy array of spin phase [rad] (1D, same length as time)
#
# Method:
#   For each spin (0→2π), fit
#       B ≈ A cos(φ) + B sin(φ)
#   by least squares, then interpolate A(t), B(t) for all times and subtract
#   the reconstructed spin tone from the raw data.

from __future__ import annotations

import numpy as np

try:
    import xarray as xr  # optional
except ImportError:  # pragma: no cover
    xr = None


# ----------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------
def _segment_by_spin_phase(phase_rad: np.ndarray,
                           wrap_threshold: float = -np.pi) -> list[tuple[int, int]]:
    """
    Split index range into spin segments using phase wrap (2π→0) detection.

    Parameters
    ----------
    phase_rad : array_like
        Spin phase [rad], shape (N,).
    wrap_threshold : float, optional
        Threshold on Δphase to detect wrap; default -π.

    Returns
    -------
    segments : list of (i_start, i_end)
        Index ranges (inclusive) for each spin.
    """
    phase_rad = np.asarray(phase_rad)
    if phase_rad.ndim != 1:
        raise ValueError("phase_rad must be 1D")

    dphi = np.diff(phase_rad)
    wrap_idx = np.where(dphi < wrap_threshold)[0]

    segments: list[tuple[int, int]] = []
    start = 0
    if len(wrap_idx) == 0:
        segments.append((0, len(phase_rad) - 1))
        return segments

    for wi in wrap_idx:
        end = wi
        if end >= start:
            segments.append((start, end))
        start = wi + 1

    if start < len(phase_rad):
        segments.append((start, len(phase_rad) - 1))

    return segments


def _fit_A_B_segment(B_seg: np.ndarray,
                     phase_seg: np.ndarray) -> tuple[float, float]:
    """
    Fit B ≈ A cos(φ) + B sin(φ) on one spin segment by least squares.
    """
    phase_seg = np.asarray(phase_seg)
    B_seg = np.asarray(B_seg)

    X = np.vstack([np.cos(phase_seg), np.sin(phase_seg)]).T  # (N, 2)
    coef, *_ = np.linalg.lstsq(X, B_seg, rcond=None)
    A, B = coef
    return float(A), float(B)


def _interp_params_to_full_time(time: np.ndarray,
                                t_fit: np.ndarray,
                                A_fit: np.ndarray,
                                B_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate spin parameters (A,B) defined at sparse fit times to all samples.

    time  : datetime64[...] array (N,)
    t_fit : datetime64[...] array (M,)
    A_fit : float array (M,)
    B_fit : float array (M,)
    """
    time = np.asarray(time)
    t_fit = np.asarray(t_fit)
    A_fit = np.asarray(A_fit)
    B_fit = np.asarray(B_fit)

    if time.ndim != 1 or t_fit.ndim != 1:
        raise ValueError("time and t_fit must be 1D")

    # convert datetime64 to seconds from first sample
    t0 = time[0].astype('datetime64[ns]')
    t_all_sec = (time - t0) / np.timedelta64(1, 's')
    t_fit_sec = (t_fit - t0) / np.timedelta64(1, 's')

    A_all = np.interp(t_all_sec, t_fit_sec, A_fit)
    B_all = np.interp(t_all_sec, t_fit_sec, B_fit)

    return A_all, B_all


# ----------------------------------------------------------------------
# core single-component & 3-component functions
# ----------------------------------------------------------------------
def remove_spintone_single_component(
    time: np.ndarray,
    B: np.ndarray,
    phase_rad: np.ndarray,
    *,
    min_points: int = 32,
    wrap_threshold: float = -np.pi,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove spin tone from one magnetic field component.

    Parameters
    ----------
    time : array_like of datetime64
        Time axis, shape (N,).
    B : array_like
        Magnetic field component, shape (N,).
    phase_rad : array_like
        Spin phase [rad], shape (N,).
    min_points : int, optional
        Minimum number of points per spin segment used for fitting.
    wrap_threshold : float, optional
        Phase jump threshold to detect spin wrap (default: -π).

    Returns
    -------
    B_clean : ndarray (N,)
        Spin-tone-removed magnetic field.
    B_spt : ndarray (N,)
        Modeled spin-tone waveform.
    A_all, B_all : ndarray (N,)
        Time series of spin-fit coefficients A(t), B(t).
    t_fit : ndarray (M,) of datetime64
        Representative times per spin used for fitting.
    A_fit, B_fit : ndarray (M,)
        Fitted A, B per spin.
    """
    time = np.asarray(time)
    B = np.asarray(B)
    phase_rad = np.asarray(phase_rad)

    if not (time.shape == B.shape == phase_rad.shape):
        raise ValueError("time, B, phase_rad must have the same shape")

    segments = _segment_by_spin_phase(phase_rad, wrap_threshold=wrap_threshold)

    t_fit_list = []
    A_fit_list = []
    B_fit_list = []

    for i_start, i_end in segments:
        if (i_end - i_start + 1) < min_points:
            continue

        idx = slice(i_start, i_end + 1)
        B_seg = B[idx]
        phase_seg = phase_rad[idx]

        A, Bc = _fit_A_B_segment(B_seg, phase_seg)

        mid_idx = (i_start + i_end) // 2
        t_fit_list.append(time[mid_idx])
        A_fit_list.append(A)
        B_fit_list.append(Bc)

    if len(t_fit_list) < 2:
        raise RuntimeError(
            "Not enough valid spin segments to interpolate A,B. "
            "Check phase_rad and min_points."
        )

    t_fit = np.array(t_fit_list)
    A_fit = np.array(A_fit_list)
    B_fit = np.array(B_fit_list)

    A_all, B_all = _interp_params_to_full_time(time, t_fit, A_fit, B_fit)

    B_spt = A_all * np.cos(phase_rad) + B_all * np.sin(phase_rad)
    B_clean = B - B_spt

    return B_clean, B_spt, A_all, B_all, t_fit, A_fit, B_fit


def remove_spintone_3comp(
    time: np.ndarray,
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray,
    phase_rad: np.ndarray,
    *,
    min_points: int = 32,
    wrap_threshold: float = -np.pi,
):
    """
    Remove spin tone from 3 components (Bx, By, Bz).

    Parameters
    ----------
    time : array_like of datetime64
        Time axis, shape (N,).
    Bx, By, Bz : array_like
        Magnetic field components, each shape (N,).
    phase_rad : array_like
        Spin phase [rad], shape (N,).
    min_points, wrap_threshold : see remove_spintone_single_component.

    Returns
    -------
    B_clean : ndarray (N, 3)
        Spin-tone-removed magnetic field [Bx, By, Bz].
    B_spt : ndarray (N, 3)
        Modeled spin-tone waveform [Bx_spt, By_spt, Bz_spt].
    params : dict
        {
            "A_all": {"x": Ax_all, "y": Ay_all, "z": Az_all},
            "B_all": {"x": Bx_all, "y": By_all, "z": Bz_all},
            "t_fit": t_fit,              # same for all components
            "A_fit": {"x": Ax_fit, ...},
            "B_fit": {"x": Bx_fit, ...},
        }
    """
    Bx_clean, Bx_spt, Ax_all, Bx_all, t_fit_x, Ax_fit, Bx_fit = \
        remove_spintone_single_component(
            time, Bx, phase_rad,
            min_points=min_points,
            wrap_threshold=wrap_threshold,
        )

    By_clean, By_spt, Ay_all, By_all, t_fit_y, Ay_fit, By_fit = \
        remove_spintone_single_component(
            time, By, phase_rad,
            min_points=min_points,
            wrap_threshold=wrap_threshold,
        )

    Bz_clean, Bz_spt, Az_all, Bz_all, t_fit_z, Az_fit, Bz_fit = \
        remove_spintone_single_component(
            time, Bz, phase_rad,
            min_points=min_points,
            wrap_threshold=wrap_threshold,
        )

    # we expect t_fit_x ≈ t_fit_y ≈ t_fit_z; use x as canonical
    B_clean = np.vstack([Bx_clean, By_clean, Bz_clean]).T
    B_spt = np.vstack([Bx_spt, By_spt, Bz_spt]).T

    params = {
        "A_all": {"x": Ax_all, "y": Ay_all, "z": Az_all},
        "B_all": {"x": Bx_all, "y": By_all, "z": Bz_all},
        "t_fit": t_fit_x,
        "A_fit": {"x": Ax_fit, "y": Ay_fit, "z": Az_fit},
        "B_fit": {"x": Bx_fit, "y": By_fit, "z": Bz_fit},
    }

    return B_clean, B_spt, params