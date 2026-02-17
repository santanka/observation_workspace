"""
CWT for E/B components in an xarray.Dataset → xarray.Dataset
- Inputs: ds: xr.Dataset with time coordinate and variables like E128_fac_x, E8_fac_x, B128_fac_y, ...
- Output: ds_cwt: xr.Dataset with spectrograms per component and COI lines

Design
- Follow tdwavelet.py logic (Morlet CWT, COI masking, gap handling, PSD-like normalization).
- Handle irregular sampling by regridding to uniform dt per input dataset.
- Small gaps (< gap_thresh) are linearly interpolated; large gaps are retained as NaN and masked in power.
- Frequencies use pywt.scale2frequency mapping. Spectrogram units are proportional to power (see Cdelta normalization).

Dependencies: numpy, pandas, xarray, pywt
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import xarray as xr
import pywt

# -----------------------------------------------------------------------------
# Core CWT on 1D time series (numpy arrays)
# -----------------------------------------------------------------------------

def _regularize_time(t, y, dt=None, gap_thresh=None):
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]; y = y[m]
    if t.size < 3:
        return np.array([]), np.array([]), np.array([], bool), np.nan, np.nan

    # 1) dtはデータから
    dt0 = float(np.median(np.diff(t)))
    if gap_thresh is None:
        gap_thresh = 5.0 * dt0

    # 等間隔グリッド
    t0 = t[0]
    idx = np.rint((t - t0) / dt0).astype(int)

    # 衝突(同じidxに複数サンプル)を解消：後勝ち
    # （交互NaNを避ける要点）
    _, last = np.unique(idx, return_index=False, return_counts=False), None
    # numpyはcountsだけ返すので dictで最後の値を採用
    sig = np.full(idx.max()+1, np.nan, float)
    for k, v in zip(idx, y): sig[k] = v

    t_grid = t0 + np.arange(sig.size) * dt0

    # 小ギャップは補間，大ギャップのみ保持
    isn = np.isnan(sig)
    if isn.any():
        # 連続NaN長さを測って「大ギャップ」を抽出
        run = np.diff(np.r_[False, isn, False]).nonzero()[0].reshape(-1,2)
        large_mask = np.zeros_like(sig, bool)
        for a,b in run:
            if (b-a)*dt0 >= gap_thresh:
                large_mask[a:b] = True
        # 小ギャップは線形補間
        sig = pd.Series(sig).interpolate(limit_area="inside").values
    else:
        large_mask = np.zeros_like(sig, bool)

    return t_grid, sig, large_mask, dt0, gap_thresh


def cwt_1d(t, y, *, dt=None, scales=None, s0=1.0, dj=1/16, J=None,
           wavelet="cmor1.5-1.0", gap_thresh=None, method="fft"):
    t_grid, sig, large_mask, dt0, gap_thresh = _regularize_time(t, y, dt, gap_thresh)
    if t_grid.size == 0:
        return t_grid, np.array([]), np.empty((0, 0), dtype=np.complex64), np.empty((0, 0), dtype=np.float32), np.array([])

    # スケール
    if scales is not None:
        S = np.asarray(list(scales), float)
    else:
        if J is None:
            J = int(np.floor(np.log2(len(sig)*dt0/s0)/dj))
        S = s0 * 2**(np.arange(J+1)*dj)

    coef, freqs = pywt.cwt(sig, S, wavelet, sampling_period=dt0, method=method)

    Cdelta = 0.776
    power = (dt0/Cdelta) * (np.abs(coef)**2)          # (F,T)

    # 2) COIは t_grid 全点で生成（NaNを作らない）
    N = t_grid.size
    time_dist = np.minimum(np.arange(N), np.arange(N)[::-1]) * dt0
    scale_coi = np.maximum(s0, (time_dist / dt0) / np.sqrt(2.0))
    freq_coi  = pywt.scale2frequency(wavelet, scale_coi) / dt0  # (T,) 全点有限

    return (
        t_grid,
        freqs,
        coef.T.astype(np.complex64, copy=False),   # (Time, Freq)
        power.T.astype(np.float32, copy=False),  # (Time, Freq)
        freq_coi.astype(np.float32, copy=False)  # (Time,)
    )


# -----------------------------------------------------------------------------
# Dataset-level API
# -----------------------------------------------------------------------------

def cwt_from_dataset(
    ds: xr.Dataset,
    *,
    dt: float | None = None,
    variables: Sequence[str] | None = None,
    s0: float = 1.0,
    dj: float = 1/16,
    J: int | None = None,
    wavelet: str = "cmor1.5-1.0",
    gap_thresh: float | None = None,
    method: str = "fft",
    suffix: str = "_cwt",
    coef_suffix: str = "_coef",
    apply_coi_mask: bool = True,
    coi_mask_side: str = "lower",
) -> xr.Dataset:
    """Run CWT for selected E/B components in `ds`.

    Selection:
      - If `variables` is None, pick variables with names starting with 'E' or 'B',
        containing '_fac_' and ending with '_x', '_y', or '_z'.

    Output variables:
      - f"{name}{suffix}": DataArray(time, freq) power
      - f"{name}_coi": DataArray(time) frequency of COI line

    Shared coordinates:
      - time (per-variable t_grid; if inputs share the same grid they will align)
      - freq_<name> per variable to avoid accidental misalignment across sampling rates
    """
    if "time" not in ds.coords:
        raise ValueError("ds must have a 'time' coordinate")

    import re
    all_vars = list(ds.data_vars)
    if variables is None:
        pat = re.compile(r"^(E|B).*_fac_.*_(x|y|z)$")
        variables = [v for v in all_vars if pat.match(v)]
    else:
        variables = [v for v in variables if v in all_vars]

    out_vars: Dict[str, xr.DataArray] = {}

    time_ref = None

    # Use numeric time axis for computation (ns to seconds float)
    t_ns = ds.time.values.astype("datetime64[ns]").astype("int64")
    t0_ns = int(t_ns[0])
    t_sec = (t_ns - t0_ns) / 1e9

    for name in variables:
        y = np.asarray(ds[name].values, dtype=float)
        t_grid, freqs, coef_tf, power_tf, freq_coi = cwt_1d(
            t_sec, y, dt=dt, s0=s0, dj=dj, J=J, wavelet=wavelet, gap_thresh=gap_thresh, method=method
        )
        if t_grid.size == 0 or freqs.size == 0:
            continue

        if apply_coi_mask:
            F = freqs[None, :]          # (1, F)
            C = freq_coi[:, None]       # (T, 1)
            if coi_mask_side == "lower":
                mask = F < C
            elif coi_mask_side == "higher":
                mask = F > C
            else:
                raise ValueError("coi_mask_side must be 'lower' or 'higher'")

            power_tf = np.where(mask, np.nan, power_tf)
            coef_tf = np.where(mask, np.nan + 1j*np.nan, coef_tf)

        # Convert back to datetime64[ns]
        time_out = np.round(t_grid * 1e9 + t0_ns).astype("int64").astype("datetime64[ns]")
        if time_ref is None:
            time_ref = time_out

        da_pow = xr.DataArray(
            power_tf,
            dims=("time", "freq"),
            coords={
                "time": time_out,
                "freq": ("freq", freqs.astype(np.float32, copy=False)),
            },
            name=f"{name}{suffix}",
            attrs={
                "long_name": f"CWT power of {name}",
                "wavelet": wavelet,
                "s0": float(s0),
                "dj": float(dj),
                "J": -1 if J is None else int(J),
                "gap_thresh": np.nan if gap_thresh is None else float(gap_thresh),
                "normalization": "(dt/Cdelta)*|coef|^2 with Cdelta=0.776",
                "units": "arbitrary power",
            },
        ).interp(time=time_ref, kwargs={"fill_value": "extrapolate"})

        da_coef = xr.DataArray(
            coef_tf,
            dims=("time", "freq"),
            coords={"time": time_out, "freq": ("freq", freqs.astype(np.float32))},
            name=f"{name}{coef_suffix}",
            attrs={
                "long_name": f"CWT power of {name}",
                "wavelet": wavelet,
                "s0": float(s0),
                "dj": float(dj),
                "J": -1 if J is None else int(J),
                "gap_thresh": np.nan if gap_thresh is None else float(gap_thresh),
                "normalization": "(dt/Cdelta)*|coef|^2 with Cdelta=0.776",
                "units": "complex coefficients",
            },
        ).interp(time=time_ref, kwargs={"fill_value": "extrapolate"})

        da_coi = xr.DataArray(
            freq_coi,
            dims=("time",),
            coords={"time": time_out},
            name=f"{name}_coi",
            attrs={
                "long_name": f"COI cutoff frequency for {name}",
                "wavelet": wavelet,
            },
        ).interp(time=time_ref, kwargs={"fill_value": "extrapolate"})

        out_vars[da_pow.name] = da_pow
        out_vars[da_coef.name] = da_coef
        out_vars[da_coi.name] = da_coi

    return xr.Dataset(out_vars).assign_coords(time=time_ref)


import numpy as np
import scipy.ndimage as ndimage

def calculate_xwt_wco(ds_cwt, var_e, var_b, dt, dj):# 1. 型の軽量化とデータの取得
    W_e = ds_cwt[var_e].values.astype(np.complex64)
    W_b = ds_cwt[var_b].values.astype(np.complex64)
    scales = (1.0 / (ds_cwt.freq.values * dt)).astype(np.float32)
    
    W_eb = W_e * np.conj(W_b)
    
    # --- 重要：NaNマスクの保持とゼロ埋め ---
    # 後で再マスクするために、元のNaNの場所を覚えておく
    # (COIマスクやギャップ部分を特定)
    mask_nan = np.isnan(W_eb)
    
    # FFTのために NaN を 0 に置換
    eb_s = np.nan_to_num(W_eb / scales)
    e2_s = np.nan_to_num((np.abs(W_e)**2) / scales)
    b2_s = np.nan_to_num((np.abs(W_b)**2) / scales)
    
    N = eb_s.shape[0]
    
    # 2. 時間軸方向にFFT
    f_eb = np.fft.fft(eb_s, axis=0)
    f_e2 = np.fft.fft(e2_s, axis=0)
    f_b2 = np.fft.fft(b2_s, axis=0)
    
    freq_fft = np.fft.fftfreq(N, d=dt).astype(np.float32)
    omega = 2 * np.pi * freq_fft
    sigmas = scales / np.sqrt(2.0)
    
    # ガウスカーネルの作成
    kernel_tf = np.exp(-0.5 * (sigmas[None, :]**2) * (omega[:, None]**2))
    
    # フィルタ適用と逆FFT
    s_eb = np.fft.ifft(f_eb * kernel_tf, axis=0)
    s_e2 = np.fft.ifft(f_e2 * kernel_tf, axis=0)
    s_b2 = np.fft.ifft(f_b2 * kernel_tf, axis=0)
    
    # 3. スケール方向の平滑化
    window_len = int(0.6 / dj) | 1  # bitwise OR
    s_eb = ndimage.uniform_filter1d(s_eb, size=window_len, axis=1)
    s_e2 = ndimage.uniform_filter1d(s_e2, size=window_len, axis=1)
    s_b2 = ndimage.uniform_filter1d(s_b2, size=window_len, axis=1)
    
    # WCO の算出
    # 分母が0になるのを防ぐための微小値 eps
    eps = 1e-100
    wco = (np.abs(s_eb)**2) / (s_e2 * s_b2 + eps)
    
    # --- 重要：再マスキング ---
    # 平滑化によってNaNだった境界が少しボケるが、元の不確実な領域をNaNに戻す
    wco[mask_nan] = np.nan
    
    phase = np.angle(W_eb, deg=True)
    
    return wco.astype(np.float32), phase.astype(np.float32)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import xarray as xr

    # Dummy example: two components at 128 Hz with small gaps
    fs = 128.0
    t = np.arange(0, 10.0, 1/fs)
    time = (t * 1e9).astype("int64").astype("datetime64[ns]")
    yx = np.sin(2*np.pi*5*t) + 0.1*np.random.randn(t.size)
    yy = np.sin(2*np.pi*12*t) + 0.1*np.random.randn(t.size)
    yx[500:520] = np.nan  # small gap
    yy[1000:1100] = np.nan  # large gap

    ds = xr.Dataset({
        "E128_fac_x": ("time", yx),
        "E128_fac_y": ("time", yy),
    }, coords={"time": time})

    ds_cwt = cwt_from_dataset(ds, dt=None, s0=1/fs, dj=1/12)
    print(ds_cwt)
