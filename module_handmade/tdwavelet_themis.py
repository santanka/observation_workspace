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
    psd_calibration: xr.DataArray | None = None,
    auto_calibrate_psd: bool = False,
    calibration_duration: float | None = None,
    calibration_freqs_ref: Sequence[float] | None = None,
    calibration_A0: float = 1.0,
    e_unit: str = "mV/m",
    b_unit: str = "nT",
) -> xr.Dataset:
    """Run CWT for selected E/B components in `ds`.

    Behavior
    --------
    - da_coef is always returned as complex wavelet coefficients.
    - If psd_calibration is None and auto_calibrate_psd is False:
        da_pow = raw wavelet power
    - If psd_calibration is provided, or auto_calibrate_psd is True:
        da_pow = calibrated, scale-rectified wavelet PSD [(unit)^2 / Hz]
    - auto_calibrate_psd は False 推奨！！！！
    """
    if "time" not in ds.coords:
        raise ValueError("ds must have a 'time' coordinate")

    # ---- optional auto calibration for PSD ----
    if psd_calibration is None and auto_calibrate_psd:
        if dt is None:
            t_ns0 = ds.time.values.astype("datetime64[ns]").astype("int64")
            dt_for_cal = float(np.median(np.diff(t_ns0))) / 1e9
        else:
            dt_for_cal = float(dt)

        fs_for_cal = 1.0 / dt_for_cal

        if calibration_duration is None:
            t0 = ds.time.values[0]
            t1 = ds.time.values[-1]
            calibration_duration = float((t1 - t0) / np.timedelta64(1, "s"))

        psd_calibration, _ = build_psd_calibration_curve(
            tw=__import__(__name__),
            fs=fs_for_cal,
            duration=calibration_duration,
            freqs_ref=calibration_freqs_ref,
            A0=calibration_A0,
            wavelet=wavelet,
            s0=s0,
            dj=dj,
            J=J,
            use_coi_mask=apply_coi_mask,
            trim_cycles=5,
        )

    import re
    all_vars = list(ds.data_vars)
    if variables is None:
        pat = re.compile(r"^(E|B).*_fac_.*_(x|y|z)$")
        variables = [v for v in all_vars if pat.match(v)]
    else:
        variables = [v for v in variables if v in all_vars]

    out_vars: Dict[str, xr.DataArray] = {}
    time_ref = None

    # numeric time axis for computation
    t_ns = ds.time.values.astype("datetime64[ns]").astype("int64")
    t0_ns = int(t_ns[0])
    t_sec = (t_ns - t0_ns) / 1e9

    use_psd_calibration = psd_calibration is not None

    for name in variables:
        y = np.asarray(ds[name].values, dtype=float)
        t_grid, freqs, coef_tf, power_tf, freq_coi = cwt_1d(
            t_sec, y, dt=dt, s0=s0, dj=dj, J=J,
            wavelet=wavelet, gap_thresh=gap_thresh, method=method
        )
        if t_grid.size == 0 or freqs.size == 0:
            continue

        if apply_coi_mask:
            F = freqs[None, :]
            C = freq_coi[:, None]
            if coi_mask_side == "lower":
                mask = F < C
            elif coi_mask_side == "higher":
                mask = F > C
            else:
                raise ValueError("coi_mask_side must be 'lower' or 'higher'")
            power_tf = np.where(mask, np.nan, power_tf)
            coef_tf = np.where(mask, np.nan + 1j*np.nan, coef_tf)

        time_out = np.round(t_grid * 1e9 + t0_ns).astype("int64").astype("datetime64[ns]")
        if time_ref is None:
            time_ref = time_out

        dt_used = float(np.median(np.diff(t_grid)))

        if name.startswith("E"):
            input_unit = e_unit
        elif name.startswith("B"):
            input_unit = b_unit
        else:
            input_unit = "unit"

        if use_psd_calibration:
            scales = _get_scale_from_freq(freqs, dt_used).astype(np.float32)
            da_scale = xr.DataArray(
                scales,
                dims=("freq",),
                coords={"freq": freqs.astype(np.float32, copy=False)},
            )
            da_K = psd_calibration.interp(freq=da_scale.freq)

            da_pow = xr.DataArray(
                power_tf,
                dims=("time", "freq"),
                coords={
                    "time": time_out,
                    "freq": ("freq", freqs.astype(np.float32, copy=False)),
                },
                name=f"{name}{suffix}",
                attrs={
                    "long_name": f"CWT PSD of {name}",
                    "wavelet": wavelet,
                    "s0": float(s0),
                    "dj": float(dj),
                    "J": -1 if J is None else int(J),
                    "gap_thresh": np.nan if gap_thresh is None else float(gap_thresh),
                    "normalization": "K_psd * ((dt/Cdelta) * |coef|^2 / scale), Cdelta=0.776",
                    "units": f"({input_unit})^2 / Hz",
                },
            )
            da_pow = (da_pow / da_scale * da_K).interp(
                time=time_ref, kwargs={"fill_value": "extrapolate"}
            )
            da_pow.name = f"{name}{suffix}"
        else:
            da_pow = xr.DataArray(
                power_tf,
                dims=("time", "freq"),
                coords={
                    "time": time_out,
                    "freq": ("freq", freqs.astype(np.float32, copy=False)),
                },
                name=f"{name}{suffix}",
                attrs={
                    "long_name": f"CWT raw power of {name}",
                    "wavelet": wavelet,
                    "s0": float(s0),
                    "dj": float(dj),
                    "J": -1 if J is None else int(J),
                    "gap_thresh": np.nan if gap_thresh is None else float(gap_thresh),
                    "normalization": "(dt/Cdelta) * |coef|^2, Cdelta=0.776",
                    "units": "raw wavelet power",
                },
            ).interp(time=time_ref, kwargs={"fill_value": "extrapolate"})
            da_pow.name = f"{name}{suffix}"

        da_coef = xr.DataArray(
            coef_tf,
            dims=("time", "freq"),
            coords={"time": time_out, "freq": ("freq", freqs.astype(np.float32))},
            name=f"{name}{coef_suffix}",
            attrs={
                "long_name": f"CWT coefficients of {name}",
                "wavelet": wavelet,
                "s0": float(s0),
                "dj": float(dj),
                "J": -1 if J is None else int(J),
                "gap_thresh": np.nan if gap_thresh is None else float(gap_thresh),
                "normalization": "coefficients from pywt.cwt",
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
                "units": "Hz",
            },
        ).interp(time=time_ref, kwargs={"fill_value": "extrapolate"})

        out_vars[da_pow.name] = da_pow
        out_vars[da_coef.name] = da_coef
        out_vars[da_coi.name] = da_coi

    return xr.Dataset(out_vars).assign_coords(time=time_ref)

def build_poynting_calibration_curve(
    tw,
    fs=64.0,
    duration=256.0,
    freqs_ref=None,
    E0=10.0,
    B0=2.0,
    phase_deg=0.0,
    wavelet="cmor1.5-1.0",
    s0=2.0,
    dj=1/32,
    J=None,
    use_coi_mask=True,
    trim_cycles=5,
):
    """
    出力周波数グリッド上に K_signed(freq), K_abs(freq) を作る。
    """
    dt = 1.0 / fs

    n = int(duration * fs)
    t = np.arange(n) * dt
    x = np.zeros_like(t)
    time = pd.to_datetime("2022-01-01") + pd.to_timedelta(t, unit="s")
    ds = xr.Dataset({"X_fac_x": ("time", x)}, coords={"time": time})

    ds_cwt = tw.cwt_from_dataset(
        ds,
        dt=dt,
        variables=["X_fac_x"],
        s0=s0,
        dj=dj,
        J=J,
        wavelet=wavelet,
        apply_coi_mask=use_coi_mask,
    )
    freqs_out = ds_cwt["freq"].values.astype(float)

    if freqs_ref is None:
        f_lo = max(freqs_out[1], 3.0 / duration)
        f_hi = min(freqs_out[-2], fs / 4.0)
        if f_hi <= f_lo:
            f_lo = freqs_out[1]
            f_hi = freqs_out[-2]
        n_ref = min(12, max(6, len(freqs_out)//40))
        freqs_ref = np.geomspace(f_lo, f_hi, n_ref)

    df_cal = calibrate_poynting_factor_scan_frequency_scaled(
        tw=tw,
        fs=fs,
        duration=duration,
        freqs_test=tuple(freqs_ref),
        E0=E0,
        B0=B0,
        phase_deg=phase_deg,
        wavelet=wavelet,
        s0=s0,
        dj=dj,
        J=J,
        use_coi_mask=use_coi_mask,
        trim_cycles=trim_cycles,
    )

    K_signed_interp = np.interp(
        freqs_out,
        df_cal["f_bin_hz"].values,
        df_cal["K_signed_for_mean_flux"].values,
        left=df_cal["K_signed_for_mean_flux"].values[0],
        right=df_cal["K_signed_for_mean_flux"].values[-1],
    )

    K_abs_interp = np.interp(
        freqs_out,
        df_cal["f_bin_hz"].values,
        df_cal["K_abs_for_absflux"].values,
        left=df_cal["K_abs_for_absflux"].values[0],
        right=df_cal["K_abs_for_absflux"].values[-1],
    )

    ds_K = xr.Dataset(
        {
            "K_signed": ("freq", K_signed_interp.astype(np.float32)),
            "K_abs": ("freq", K_abs_interp.astype(np.float32)),
        },
        coords={"freq": freqs_out.astype(np.float32)},
    )
    return ds_K, df_cal


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
    #sigmas = scales / np.sqrt(2.0)
    
    # ガウスカーネルの作成
    #kernel_tf = np.exp(-0.5 * (sigmas[None, :]**2) * (omega[:, None]**2))
    sigmas_t = (scales * dt / np.sqrt(2.0)).astype(np.float32)
    kernel_tf = np.exp(-0.5 * (omega[:, None]**2) * (sigmas_t[None, :]**2))

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

import numpy as np
import scipy.ndimage as ndimage

def calculate_xwt_wco_blocked(ds_cwt, var_e, var_b, dt, dj, block_f=32):
    W_e = ds_cwt[var_e].values.astype(np.complex64, copy=False)
    W_b = ds_cwt[var_b].values.astype(np.complex64, copy=False)
    scales = (1.0 / (ds_cwt.freq.values * dt)).astype(np.float32)

    W_eb = W_e * np.conj(W_b)
    mask_nan = np.isnan(W_eb)

    # NaN -> 0（時間平滑はFFTでやるので）
    eb_s_all = np.nan_to_num(W_eb / scales)
    e2_s_all = np.nan_to_num((np.abs(W_e)**2) / scales)
    b2_s_all = np.nan_to_num((np.abs(W_b)**2) / scales)

    N, F = eb_s_all.shape
    freq_fft = np.fft.fftfreq(N, d=dt).astype(np.float32)
    omega = (2*np.pi*freq_fft).astype(np.float32)

    # 時間平滑後の入れ物（これだけは最終的に必要）
    s_eb = np.empty((N, F), dtype=np.complex64)
    s_e2 = np.empty((N, F), dtype=np.complex64)
    s_b2 = np.empty((N, F), dtype=np.complex64)

    for f0 in range(0, F, block_f):
        f1 = min(F, f0 + block_f)
        sigmas = (scales[f0:f1] / np.sqrt(2.0)).astype(np.float32)  # (Fb,)

        # ブロックだけFFT
        f_eb = np.fft.fft(eb_s_all[:, f0:f1], axis=0)
        f_e2 = np.fft.fft(e2_s_all[:, f0:f1], axis=0)
        f_b2 = np.fft.fft(b2_s_all[:, f0:f1], axis=0)

        # kernel もブロックだけ作る（巨大行列を避ける）
        kernel = np.exp(-0.5 * (omega[:, None]**2) * (sigmas[None, :]**2)).astype(np.float32)

        s_eb[:, f0:f1] = np.fft.ifft(f_eb * kernel, axis=0).astype(np.complex64, copy=False)
        s_e2[:, f0:f1] = np.fft.ifft(f_e2 * kernel, axis=0).astype(np.complex64, copy=False)
        s_b2[:, f0:f1] = np.fft.ifft(f_b2 * kernel, axis=0).astype(np.complex64, copy=False)

        # ここで f_eb/f_e2/f_b2/kernel はスコープ外で破棄される

    # スケール方向の平滑（ここは全周波数一括）
    window_len = int(0.6 / dj) | 1
    s_eb = ndimage.uniform_filter1d(s_eb, size=window_len, axis=1)
    s_e2 = ndimage.uniform_filter1d(s_e2, size=window_len, axis=1)
    s_b2 = ndimage.uniform_filter1d(s_b2, size=window_len, axis=1)

    eps = 1e-100
    wco = (np.abs(s_eb)**2) / (s_e2 * s_b2 + eps)
    wco[mask_nan] = np.nan
    phase = np.angle(W_eb, deg=True)

    return wco.astype(np.float32), phase.astype(np.float32)

import numpy as np
import scipy.ndimage as ndimage

def wco_hist_streaming_from_cwtcoef(
    W_e: np.ndarray, W_b: np.ndarray,
    dt: float, dj: float, freqs: np.ndarray,
    n_points: int, bins: np.ndarray,
    block_f: int = 32, eps: float = 1e-100
) -> np.ndarray:
    """
    W_e, W_b: (N, F) complex64 (coef)
    freqs: (F,) float
    bins: 1D bin edges (e.g., np.linspace(0,1,1001))
    return: total_hists (F, n_bins)
    """

    W_e = np.asarray(W_e, np.complex64)
    W_b = np.asarray(W_b, np.complex64)
    freqs = np.asarray(freqs, np.float32)

    N, F = W_e.shape
    assert W_b.shape == (N, F)

    n_bins = len(bins) - 1
    total_h = np.zeros((F, n_bins), dtype=np.int64)

    # time cut: remove edge quarters (your current design)
    mid = n_points // 4
    t0, t1 = mid, N - mid
    if t1 <= t0 + 1:
        return total_h

    # scale axis
    scales = (1.0 / (freqs * dt)).astype(np.float32)

    # smoothing in scale axis (must match your calculate_xwt_wco)
    window_len = int(0.6 / dj)
    if window_len % 2 == 0:
        window_len += 1
    halo = window_len // 2

    # omega for time smoothing
    omega = (2.0 * np.pi * np.fft.fftfreq(N, d=dt)).astype(np.float32)

    # For hist binning in [0,1]
    inv_bw = n_bins  # since bins assumed uniform on [0,1], bin = floor(wco*n_bins)

    for f0 in range(0, F, block_f):
        f1 = min(F, f0 + block_f)

        # haloed band for correct scale smoothing
        a0 = max(0, f0 - halo)
        a1 = min(F, f1 + halo)

        We = W_e[:, a0:a1]
        Wb = W_b[:, a0:a1]
        sc = scales[a0:a1]  # (Fb_halo,)

        Web = We * np.conj(Wb)
        mask_nan = np.isnan(Web)

        # NaN->0 for FFT smoothing
        eb_s = np.nan_to_num(Web / sc, nan=0.0)
        e2_s = np.nan_to_num((np.abs(We) ** 2) / sc, nan=0.0)
        b2_s = np.nan_to_num((np.abs(Wb) ** 2) / sc, nan=0.0)

        sigmas = (sc / np.sqrt(2.0)).astype(np.float32)  # (Fb_halo,)
        kernel = np.exp(-0.5 * (omega[:, None] ** 2) * (sigmas[None, :] ** 2)).astype(np.float32)

        # time smoothing
        s_eb = np.fft.ifft(np.fft.fft(eb_s, axis=0) * kernel, axis=0).astype(np.complex64, copy=False)
        s_e2 = np.fft.ifft(np.fft.fft(e2_s, axis=0) * kernel, axis=0).astype(np.complex64, copy=False)
        s_b2 = np.fft.ifft(np.fft.fft(b2_s, axis=0) * kernel, axis=0).astype(np.complex64, copy=False)

        # scale smoothing (on haloed band)
        s_eb = ndimage.uniform_filter1d(s_eb, size=window_len, axis=1)
        s_e2 = ndimage.uniform_filter1d(s_e2, size=window_len, axis=1)
        s_b2 = ndimage.uniform_filter1d(s_b2, size=window_len, axis=1)

        wco = (np.abs(s_eb) ** 2) / (s_e2 * s_b2 + eps)
        wco = wco.astype(np.float32)
        wco[mask_nan] = np.nan

        # pick central band only (drop halo)
        c0 = f0 - a0
        c1 = c0 + (f1 - f0)
        wco_c = wco[t0:t1, c0:c1]  # (Nt, Fb)

        # ---- fast histogram update (no per-freq Python loop) ----
        # reshape to (Fb, Nt) for easier freq indexing
        w = wco_c.T  # (Fb, Nt)
        valid = np.isfinite(w)
        if not valid.any():
            continue

        # clip to [0,1] since bins are defined there
        ww = np.clip(w, 0.0, 1.0)
        bin_idx = np.floor(ww * inv_bw).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # flatten valid points
        bin_flat = bin_idx[valid]
        # freq index per element
        freq_ids = (f0 + np.arange(f1 - f0, dtype=np.int32))[:, None]
        freq_flat = np.broadcast_to(freq_ids, w.shape)[valid]

        np.add.at(total_h, (freq_flat, bin_flat), 1)

    return total_h

import numpy as np
import pandas as pd
import xarray as xr

MU0 = 4e-7 * np.pi


def _get_scale_from_freq(freq_hz: np.ndarray, dt: float) -> np.ndarray:
    """
    君の tw.calculate_xwt_wco と整合する scale 定義。
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    return 1.0 / (freq_hz * dt)


def calibrate_poynting_factor_single_tone_scaled(
    tw,
    fs=64.0,
    duration=256.0,
    f0=0.2,
    E0=1.0,
    B0=1.0,
    phase_deg=0.0,
    wavelet="cmor1.5-1.0",
    s0=2.0,
    dj=1/32,
    J=None,
    use_coi_mask=True,
    trim_cycles=5,
):
    """
    単色波を使って、proxy/scale 版の S_parallel 較正係数を求める。

    ここでの proxy は
        proxy_scaled = Re(WE1*conj(WB2) - WE2*conj(WB1)) / scale
    とする。
    """

    dt = 1.0 / fs
    n = int(duration * fs)
    t = np.arange(n) * dt
    phase = np.deg2rad(phase_deg)

    # 右手系 (x, y, z=b0) を想定
    # S_parallel = (E_x B_y - E_y B_x) / mu0
    Ex = E0 * np.cos(2 * np.pi * f0 * t)
    Ey = np.zeros_like(Ex)
    Bx = np.zeros_like(Ex)
    By = B0 * np.cos(2 * np.pi * f0 * t + phase)

    time = pd.to_datetime("2022-01-01") + pd.to_timedelta(t, unit="s")
    ds = xr.Dataset(
        {
            "E64_fac_x": ("time", Ex),
            "E64_fac_y": ("time", Ey),
            "B64_fac_x": ("time", Bx),
            "B64_fac_y": ("time", By),
        },
        coords={"time": time},
    )

    # ここでは raw CWT で十分。da_coef が欲しいだけなので PSD 較正は不要。
    ds_cwt = tw.cwt_from_dataset(
        ds,
        dt=dt,
        variables=["E64_fac_x", "E64_fac_y", "B64_fac_x", "B64_fac_y"],
        s0=s0,
        dj=dj,
        J=J,
        wavelet=wavelet,
        apply_coi_mask=use_coi_mask,
        psd_calibration=None,
        auto_calibrate_psd=False,
    )

    freqs = ds_cwt["freq"].values.astype(float)
    i_f = int(np.argmin(np.abs(freqs - f0)))
    f_bin = float(freqs[i_f])
    scale_bin = float(_get_scale_from_freq(np.array([f_bin]), dt)[0])

    WE1 = ds_cwt["E64_fac_x_coef"].values[:, i_f]
    WE2 = ds_cwt["E64_fac_y_coef"].values[:, i_f]
    WB1 = ds_cwt["B64_fac_x_coef"].values[:, i_f]
    WB2 = ds_cwt["B64_fac_y_coef"].values[:, i_f]

    cross = np.real(WE1 * np.conj(WB2) - WE2 * np.conj(WB1))
    proxy_scaled = cross / scale_bin

    # ---- trim_cycles を安全側に自動調整 ----
    # 前後で全長の 45% を超えないようにする
    max_trim_cycles = max(0.0, 0.45 * duration * f_bin)
    trim_cycles_eff = min(float(trim_cycles), max_trim_cycles)

    n_trim = int(trim_cycles_eff * fs / f_bin)

    # 念のため最終防衛
    max_n_trim = max(0, (len(proxy_scaled) - 2) // 2)
    n_trim = min(n_trim, max_n_trim)

    if n_trim > 0:
        sl = slice(n_trim, len(proxy_scaled) - n_trim)
    else:
        sl = slice(0, len(proxy_scaled))

    proxy_valid = proxy_scaled[sl]
    proxy_valid = proxy_valid[np.isfinite(proxy_valid)]

    if proxy_valid.size == 0:
        raise RuntimeError(
            f"No valid proxy values remain after masking/trimming. "
            f"f0_input={f0:.6g}, f_bin={f_bin:.6g}, duration={duration:.6g}, "
            f"trim_cycles_eff={trim_cycles_eff:.3f}"
        )

    proxy_mean = float(np.nanmean(proxy_valid))
    proxy_abs_mean = float(np.nanmean(np.abs(proxy_valid)))

    # 理論的な時間平均 Poynting flux
    # <S> = (1 / (2 mu0)) E0 B0 cos(phi)
    S_theory_mean = (E0 * B0 * np.cos(phase)) / (2.0 * MU0)

    # 参考: 時系列から直接計算した instantaneous S
    S_inst = (Ex * By - Ey * Bx) / MU0
    S_inst_mean = float(np.mean(S_inst[sl]))
    S_inst_abs_mean = float(np.mean(np.abs(S_inst[sl])))

    K_signed = S_theory_mean / proxy_mean if proxy_mean != 0 else np.nan
    K_abs = S_inst_abs_mean / proxy_abs_mean if proxy_abs_mean != 0 else np.nan

    result = {
        "f0_input_hz": f0,
        "f_bin_hz": f_bin,
        "scale_bin": scale_bin,
        "phase_deg": phase_deg,
        "trim_cycles_requested": float(trim_cycles),
        "trim_cycles_effective": float(trim_cycles_eff),
        "n_trim": int(n_trim),
        "S_theory_mean": S_theory_mean,
        "S_inst_mean_from_timeseries": S_inst_mean,
        "S_inst_abs_mean_from_timeseries": S_inst_abs_mean,
        "proxy_scaled_mean": proxy_mean,
        "proxy_scaled_abs_mean": proxy_abs_mean,
        "K_signed_for_mean_flux": K_signed,
        "K_abs_for_absflux": K_abs,
        "suggested_formula_signed": (
            "Spara_tf ~= K_signed * Re(WE1*conj(WB2) - WE2*conj(WB1)) / scale"
        ),
        "suggested_formula_abs": (
            "abs_Spara_tf ~= K_abs * abs(Re(WE1*conj(WB2) - WE2*conj(WB1)) / scale)"
        ),
    }

    return result, ds_cwt


def calibrate_poynting_factor_scan_frequency_scaled(
    tw,
    fs=64.0,
    duration=256.0,
    freqs_test=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    E0=1.0,
    B0=1.0,
    phase_deg=0.0,
    wavelet="cmor1.5-1.0",
    s0=2.0,
    dj=1/32,
    J=None,
    use_coi_mask=True,
    trim_cycles=5,
):
    """
    proxy/scale 版で複数周波数の較正係数を調べる。
    """
    rows = []
    for f0 in freqs_test:
        result, _ = calibrate_poynting_factor_single_tone_scaled(
            tw=tw,
            fs=fs,
            duration=duration,
            f0=f0,
            E0=E0,
            B0=B0,
            phase_deg=phase_deg,
            wavelet=wavelet,
            s0=s0,
            dj=dj,
            J=J,
            use_coi_mask=use_coi_mask,
            trim_cycles=trim_cycles,
        )
        rows.append(result)

    return pd.DataFrame(rows)

import numpy as np
import pandas as pd
import xarray as xr

def _get_scale_from_freq(freq_hz: np.ndarray, dt: float) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=float)
    return 1.0 / (freq_hz * dt)


def calibrate_auto_psd_factor_single_tone_scaled(
    tw,
    fs=64.0,
    duration=256.0,
    f0=0.2,
    A0=1.0,
    wavelet="cmor1.5-1.0",
    s0=2.0,
    dj=1/32,
    J=None,
    use_coi_mask=True,
    trim_cycles=5,
    variable_name="X_fac_x",
):
    """
    単色波 x(t)=A0*cos(2πf0 t) を入力し、
    rectified wavelet power /Hz への較正係数 K_psd を求める。

    ここで rectified power は
        P_rect = power / scale
    とする。

    出力 K_psd は
        PSD_tf ~= K_psd * P_rect
    となるように定める。
    """

    dt = 1.0 / fs
    n = int(duration * fs)
    t = np.arange(n) * dt
    x = A0 * np.cos(2 * np.pi * f0 * t)

    time = pd.to_datetime("2022-01-01") + pd.to_timedelta(t, unit="s")
    ds = xr.Dataset(
        {variable_name: ("time", x)},
        coords={"time": time},
    )

    ds_cwt = cwt_from_dataset(
        ds,
        dt=dt,
        variables=[variable_name],
        s0=s0,
        dj=dj,
        J=J,
        wavelet=wavelet,
        apply_coi_mask=use_coi_mask,
        psd_calibration=None,
        auto_calibrate_psd=False,
    )

    freqs = ds_cwt["freq"].values.astype(float)
    i_f = int(np.argmin(np.abs(freqs - f0)))
    f_bin = float(freqs[i_f])

    df = np.abs(np.gradient(freqs))
    df_bin = float(df[i_f])

    scales = _get_scale_from_freq(freqs, dt)
    scale_bin = float(scales[i_f])

    power_bin = ds_cwt[f"{variable_name}_cwt"].values[:, i_f].astype(float)
    rect_bin = power_bin / scale_bin

    # ---- trim_cycles を安全側に自動調整 ----
    max_trim_cycles = max(0.0, 0.45 * duration * f_bin)
    trim_cycles_eff = min(float(trim_cycles), max_trim_cycles)

    n_trim = int(trim_cycles_eff * fs / f_bin)

    max_n_trim = max(0, (len(rect_bin) - 2) // 2)
    n_trim = min(n_trim, max_n_trim)

    if n_trim > 0:
        sl = slice(n_trim, len(rect_bin) - n_trim)
    else:
        sl = slice(0, len(rect_bin))

    rect_valid = rect_bin[sl]
    rect_valid = rect_valid[np.isfinite(rect_valid)]

    if rect_valid.size == 0:
        raise RuntimeError(
            f"No valid rectified power values remain after masking/trimming. "
            f"f0_input={f0:.6g}, f_bin={f_bin:.6g}, duration={duration:.6g}, "
            f"trim_cycles_eff={trim_cycles_eff:.3f}"
        )

    rect_mean = float(np.nanmean(rect_valid))

    # one-sided PSD のビン積分が A0^2/2 になるようにする
    psd_theory_peak = (A0**2 / 2.0) / df_bin
    K_psd = psd_theory_peak / rect_mean if rect_mean != 0 else np.nan

    return {
        "f0_input_hz": f0,
        "f_bin_hz": f_bin,
        "df_bin_hz": df_bin,
        "scale_bin": scale_bin,
        "trim_cycles_requested": float(trim_cycles),
        "trim_cycles_effective": float(trim_cycles_eff),
        "n_trim": int(n_trim),
        "rectified_power_mean": rect_mean,
        "psd_theory_peak": psd_theory_peak,
        "K_psd": K_psd,
    }, ds_cwt


def calibrate_auto_psd_factor_scan_frequency_scaled(
    tw,
    fs=64.0,
    duration=256.0,
    freqs_test=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    A0=1.0,
    wavelet="cmor1.5-1.0",
    s0=2.0,
    dj=1/32,
    J=None,
    use_coi_mask=True,
    trim_cycles=5,
    variable_name="X_fac_x",
):
    rows = []
    for f0 in freqs_test:
        result, _ = calibrate_auto_psd_factor_single_tone_scaled(
            tw=tw,
            fs=fs,
            duration=duration,
            f0=f0,
            A0=A0,
            wavelet=wavelet,
            s0=s0,
            dj=dj,
            J=J,
            use_coi_mask=use_coi_mask,
            trim_cycles=trim_cycles,
            variable_name=variable_name,
        )
        rows.append(result)
    return pd.DataFrame(rows)


def build_psd_calibration_curve(
    tw,
    fs=64.0,
    duration=256.0,
    freqs_ref=None,
    A0=1.0,
    wavelet="cmor1.5-1.0",
    s0=2.0,
    dj=1/32,
    J=None,
    use_coi_mask=True,
    trim_cycles=5,
):
    """
    出力周波数グリッド上に K_psd(freq) を作る。
    """
    dt = 1.0 / fs

    # ダミー系列で freq グリッドを取得
    n = int(duration * fs)
    t = np.arange(n) * dt
    x = np.zeros_like(t)
    time = pd.to_datetime("2022-01-01") + pd.to_timedelta(t, unit="s")
    ds = xr.Dataset({"X_fac_x": ("time", x)}, coords={"time": time})

    ds_cwt = cwt_from_dataset(
        ds,
        dt=dt,
        variables=["X_fac_x"],
        s0=s0,
        dj=dj,
        J=J,
        wavelet=wavelet,
        apply_coi_mask=use_coi_mask,
    )
    freqs_out = ds_cwt["freq"].values.astype(float)

    if freqs_ref is None:
        # 出力ビンの中から数点選ぶ
        idx = np.unique(
            np.round(np.linspace(1, len(freqs_out)-2, min(12, len(freqs_out)-2))).astype(int)
        )
        freqs_ref = freqs_out[idx]

    df_cal = calibrate_auto_psd_factor_scan_frequency_scaled(
        tw=tw,
        fs=fs,
        duration=duration,
        freqs_test=tuple(freqs_ref),
        A0=A0,
        wavelet=wavelet,
        s0=s0,
        dj=dj,
        J=J,
        use_coi_mask=use_coi_mask,
        trim_cycles=trim_cycles,
        variable_name="X_fac_x",
    )

    K_interp = np.interp(
        freqs_out,
        df_cal["f_bin_hz"].values,
        df_cal["K_psd"].values,
        left=df_cal["K_psd"].values[0],
        right=df_cal["K_psd"].values[-1],
    )

    da_K = xr.DataArray(
        K_interp.astype(np.float32),
        dims=("freq",),
        coords={"freq": freqs_out.astype(np.float32)},
        name="K_psd",
        attrs={
            "long_name": "Calibration factor for scale-corrected wavelet PSD",
            "units": "depends on input units; output becomes input_unit^2/Hz",
        },
    )

    return da_K, df_cal


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
