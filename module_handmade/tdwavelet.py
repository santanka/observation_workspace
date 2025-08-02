# tdwavelet.py — Continuous Wavelet Transform helper (Morlet) + COI overlay
"""
Refined using the labo‑code.com recipe and now **draws the Cone‑of‑Influence
(COI)** on the same spectrogram so edge‑effect regions are clearly visible
(just like the MATLAB/IDL examples in Torrence & Compo, 1998).

Usage snippet
-------------
>>> tdwavelet('erg_mag_fac_x_norm_hp', dt=1/64)
>>> pt.tplot(['erg_mag_fac_x_norm_hp_cwt',   # spectrogram
...           'erg_mag_fac_x_norm_hp_coi']) # COI line overlay

Implementation notes
--------------------
* COI half‑width = √2 · scale [s] (T&C Eq.(11)).
* We convert that to a **cut‑off frequency line** `f_coi(t) = f_fourier(scale_coi)`
  and store it as a normal 2‑column tplot variable so it overlays naturally.
* Everything else (gap handling, explicit/automatic scales, PSD normalisation)
  is unchanged from the previous version.
"""
from __future__ import annotations

from typing import List, Sequence, Union, Iterable
import numpy as np
import pandas as pd
import pywt, pyspedas as psp, pytplot as pt

__all__ = ["tdwavelet"]

# -----------------------------------------------------------------------------
# helper
# -----------------------------------------------------------------------------

def _as_list(v):
    return list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v]

# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def tdwavelet(
    var_names: Union[str, Sequence[str]],
    *,
    dt: float | None = None,
    scales: Iterable[float] | None = None,
    s0: float = 1.0,
    dj: float = 1/12,
    J: int | None = None,
    wavelet: str = "cmor1.5-1.0",
    gap_thresh: float | None = None,
    method: str = "fft",
    suffix: str = "_cwt",
    overwrite: bool = True,
    zrange: Sequence[float] | None = None,
    **pt_opts,
) -> List[str]:
    """Compute Morlet CWT + COI line and store both as tplot variables."""

    outs: List[str] = []
    for vin in _as_list(var_names):
        t, data = psp.get_data(vin)
        if t is None or data is None:
            print(f"tdwavelet: '{vin}' not found, skip")
            continue

        good = np.isfinite(data)
        t, data = t[good], data[good]
        if len(t) < 4:
            continue

        dt0 = float(dt) if dt else float(np.median(np.diff(t)))
        if gap_thresh is None:
            gap_thresh = 5*dt0

        # regular grid via integer indices
        idx  = np.round((t - t[0]) / dt0).astype(int)
        N    = idx[-1] + 1
        t_grid = t[0] + np.arange(N)*dt0
        sig = np.full(N, np.nan)
        sig[idx] = data

        isnan = np.isnan(sig)
        if isnan.any():
            run_id = np.cumsum(np.concatenate(([0], np.diff(isnan))))
            run_len = np.bincount(run_id)[run_id]
            small = isnan & (run_len*dt0 < gap_thresh)
            if small.any():
                sig = pd.Series(sig).mask(small).interpolate(limit_area='inside').values
            large_mask = np.isnan(sig)
            sig = pd.Series(sig).ffill().bfill().values
        else:
            large_mask = np.zeros_like(sig, dtype=bool)

        # scales
        if scales is not None:
            S = np.asarray(list(scales), float)
        else:
            if J is None:
                J = int(np.floor(np.log2(len(sig)*dt0/s0)/dj))
            S = s0 * 2**(np.arange(J+1)*dj)

        coef, freqs = pywt.cwt(sig, S, wavelet, sampling_period=dt0, method=method)
        Cdelta = 0.776
        power = (dt0 / (Cdelta)) * (np.abs(coef)**2)

        # ---------------- COI calculation --------------------------------
        time_dist = np.minimum(np.arange(N), np.arange(N)[::-1]) * dt0
        scale_coi = np.maximum(s0, time_dist/np.sqrt(2))
        freq_coi  = pywt.scale2frequency(wavelet, scale_coi) / dt0

        # COI周波数を下回る要素をマスクする
        # freqs を列ベクトルに、freq_coi を行ベクトルとして比較し、2Dマスクを作成
        coi_mask = freqs[:, None] < freq_coi
        power[coi_mask] = np.nan

        # 既存の大きなギャップのマスクも適用
        power[:, large_mask] = np.nan
        power = power.T

        # store power spectrogram
        vout = vin + suffix
        if overwrite and vout in pt.data_quants:
            pt.del_data(vout)
        psp.store_data(vout, data={'x': t_grid, 'y': power, 'v': freqs})
        pt.options(vout, 'spec', True)
        pt.options(vout, 'ylog', pt_opts.get('ylog', 1))
        pt.options(vout, 'zlog', pt_opts.get('zlog', 1))
        pt.options(vout, 'colormap', pt_opts.get('colormap', 'turbo'))
        if zrange:
            pt.options(vout, 'zrange', list(zrange))
        pt.options(vout, 'ytitle', f"{vin} (CWT)")
        outs.append(vout)

        # store COI line (non‑log y‑axis so plot overlays)
        vcoi = vin + '_coi'
        if overwrite and vcoi in pt.data_quants:
            pt.del_data(vcoi)
        psp.store_data(vcoi, data={'x': t_grid, 'y': freq_coi})
        pt.options(vcoi, 'thick', 2)
        pt.options(vcoi, 'color', 'white')
        pt.options(vcoi, 'legend_names', ['COI'])
        outs.append(vcoi)

    return outs
