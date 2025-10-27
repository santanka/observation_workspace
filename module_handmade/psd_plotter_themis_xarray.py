# psd_plotter_themis_xarray.py
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 周波数ごと結合（timeは結合、freqは必要帯のみ） ----------
def _concat_and_slice(dsets, var, fmin=None, fmax=None):
    das = [ds[var] for ds in dsets if var in ds]
    if not das:
        return None
    da = xr.concat(das, dim="time").sortby("time")
    if fmin is not None or fmax is not None:
        lo = da.freq.min().item() if fmin is None else fmin
        hi = da.freq.max().item() if fmax is None else fmax
        da = da.sel(freq=slice(lo, hi))
    return da

def _make_unique_time(obj):
    obj = obj.sortby("time")
    if not pd.Index(obj.time.values).has_duplicates:
        return obj
    return obj.groupby("time").mean()

# ---------- データ辞書構築（8Hz帯と128Hz帯を分離管理） ----------
def build_data_dict_xr(dsets_8_clean, dsets_128, ds_velocity_ms, ds_parameter,
                       f_split=4.0, cutoff_freq=None):
    dd = {}
    # PSD: 8Hz帯(<=f_split), 128Hz帯(>=f_split)
    for comp in ["x","y"]:
        dd[f"E{comp}_8"]   = _concat_and_slice(dsets_8_clean,   f"E8_fac_{comp}_cwt_clean", fmin=None, fmax=f_split)
        dd[f"E{comp}_128"] = _concat_and_slice(dsets_128,      f"E128_fac_{comp}_cwt",      fmin=f_split, fmax=None)
        dd[f"B{comp}_8"]   = _concat_and_slice(dsets_8_clean,   f"B8_fac_{comp}_cwt_clean", fmin=None, fmax=f_split)
        dd[f"B{comp}_128"] = _concat_and_slice(dsets_128,      f"B128_fac_{comp}_cwt",      fmin=f_split, fmax=None)

    # 補助量: vel基準で時間整列
    vel_u = _make_unique_time(ds_velocity_ms)
    par_u = _make_unique_time(ds_parameter)
    t = vel_u.time

    dd["v_A_"]           = vel_u["Alfven_speed"]                 # m/s
    dd["ion_cyclo_freq"] = par_u["proton_cycl_freq_Hz"].interp(time=t)
    dd["V_ion_fac_perp"] = vel_u["perp_sys_speed"]
    dd["V_th_proton"]    = vel_u["ion_thermal_speed"]
    dd["C_s_proton"]     = vel_u["ion_acoustic_speed"]

    dd["tau"]    = (dd["V_th_proton"]/dd["C_s_proton"])**2 / 2.0
    dd["beta_i"] = (dd["V_th_proton"]/dd["v_A_"])**2
    dd["r_param"]= dd["C_s_proton"]/dd["V_th_proton"]

    dd["cutoff_freq"] = None if cutoff_freq is None else np.asarray(cutoff_freq, float)
    return dd

def _keep_mask_by_cutoff(freq: np.ndarray, cutoff):
    if cutoff is None:
        return np.ones_like(freq, dtype=bool)
    c = np.asarray(cutoff, float)
    if c.size == 2:
        mask = (freq <= c[0]) | (freq >= c[1])
    elif c.size == 3:
        mask = ((freq >= c[0]) & (freq <= c[1])) | (freq >= c[2])
    elif c.size == 4:
        mask = ((freq >= c[0]) & (freq <= c[3]))
        #mask = ((freq >= c[0]) & (freq <= c[1])) | ((freq >= c[2]) & (freq <= c[3]))
    else:
        mask = freq >= c.item()
    return mask  # True=描く, False=隠す（NaNにする）

# 置換：スピントーン固定マスク → cutoff対応マスク
def _apply_cutoff_mask(freq, arr, cutoff):
    keep = _keep_mask_by_cutoff(freq, cutoff)
    out = np.array(arr, copy=True)
    out[~keep] = np.nan
    return out

def _pick_window_time(dd, t0, t1):
    """窓内にサンプルを持つ time をどれかの帯から拾う。無ければ None"""
    for key in ("Bx_8","Bx_128","Ex_8","Ex_128","By_8","By_128","Ey_8","Ey_128"):
        da = dd.get(key)
        if da is None:
            continue
        try:
            da_sel = da.sel(time=slice(t0, t1))
        except Exception:
            continue
        if 'time' not in da_sel.coords:
            continue
        ti = da_sel.coords['time']
        if getattr(ti, "size", 0) > 0:
            return ti
    return None


# ---------- 周波数スペクトル（8帯/128帯を独立に平均し重ね描画） ----------
def plot_freq_spectrum_dual(dd, t0, dt_sec=60):
    # --- 元コードと同じ外観 ---
    plt.rcdefaults()
    plt.rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    t1 = t0 + np.timedelta64(dt_sec, "s")

    # 補助量平均
    interp_time = _pick_window_time(dd, t0, t1)
    if interp_time is None or interp_time.size == 0:
        plt.close(fig)
        return None

    try:
        vA   = dd["v_A_"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        ionf = dd["ion_cyclo_freq"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Vp   = dd["V_ion_fac_perp"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Vth  = dd["V_th_proton"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Cs   = dd["C_s_proton"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        tau  = dd["tau"].interp(time=interp_time).sel(time=slice(t0, t1)).values
        beta = dd["beta_i"].interp(time=interp_time).sel(time=slice(t0, t1)).values
    except ValueError:
        plt.close(fig)
        return None

    # バンド別の統計計算
    def band_stats(band):
        Bx = dd[f'Bx_{band}'].sel(time=slice(t0, t1)).values
        By = dd[f'By_{band}'].sel(time=slice(t0, t1)).values
        Ex = dd[f'Ex_{band}'].sel(time=slice(t0, t1)).values
        Ey = dd[f'Ey_{band}'].sel(time=slice(t0, t1)).values
        if Bx.size == 0: return None
        mBx, sBx = np.nanmean(Bx, 0), np.nanstd(Bx, 0)
        mBy, sBy = np.nanmean(By, 0), np.nanstd(By, 0)
        mEx, sEx = np.nanmean(Ex, 0), np.nanstd(Ex, 0)
        mEy, sEy = np.nanmean(Ey, 0), np.nanstd(Ey, 0)
        Bp2    = (mBx + mBy) * 1e-18          # nT^2→T^2
        Bp2err = np.sqrt(sBx**2 + sBy**2) * 1e-18
        Ep2    = (mEx + mEy)                   # (mV/m)^2/Hz
        Ep2err = np.sqrt(sEx**2 + sEy**2)
        f = dd[f'Ex_{band}'].freq.values
        return f, Bp2, Bp2err, Ep2, Ep2err
    
    cf = dd.get('cutoff_freq')

    # 上段: PSD（電場=橙, 磁場=青。8/128で色は変えない）
    for band in ('8', '128'):
        st = band_stats(band)
        if st is None: continue
        f, Bp2, Bp2err, Ep2, Ep2err = st
        VA2Bp2 = (vA**2) * Bp2 * 1e6              # (V/m)^2→(mV/m)^2
        VA2err = (vA**2) * Bp2err * 1e6
        VA2Bp2 = _apply_cutoff_mask(f, VA2Bp2, cf)
        VA2err = _apply_cutoff_mask(f, VA2err, cf)
        Ep2    = _apply_cutoff_mask(f, Ep2,    cf)
        Ep2err = _apply_cutoff_mask(f, Ep2err, cf)
        if band == '8':
            ax1.errorbar(f, VA2Bp2, yerr=VA2err, fmt='o', ms=3, color='b',      label=r'$v_\mathrm{A}^{2}B_\perp^{2}$ PSD')
            ax1.errorbar(f, Ep2,    yerr=Ep2err, fmt='o', ms=3, color='orange', label=r'$E_\perp^{2}$ PSD')
        else:
            ax1.errorbar(f, VA2Bp2, yerr=VA2err, fmt='o', ms=3, color='b')
            ax1.errorbar(f, Ep2,    yerr=Ep2err, fmt='o', ms=3, color='orange')

    # 下段: ratio（スピントーン帯は未描画）
    for band in ('8', '128'):
        st = band_stats(band)
        if st is None: continue
        f, Bp2, Bp2err, Ep2, Ep2err = st
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.sqrt(Ep2*1e-6 / Bp2) / vA
            relE2 = (Ep2err / Ep2)**2
            relB2 = (Bp2err / Bp2)**2
            rerr  = ratio * np.sqrt(0.25*relE2 + 0.25*relB2)
        ratio  = _apply_cutoff_mask(f, ratio,  cf)
        rerr   = _apply_cutoff_mask(f, rerr,   cf)
        if band == '8':
            ax2.errorbar(f, ratio, yerr=rerr, fmt='o', ms=3, color='k', ls='', label=r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$')
        else:
            ax2.errorbar(f, ratio, yerr=rerr, fmt='o', ms=3, color='k', ls='')

    # 理論曲線（V–M）
    fth = np.logspace(-2, 2, 1000)
    tau_gm  = np.nanmedian(tau); beta_gm = np.nanmedian(beta)
    VB_med = (1 + (fth/ionf*Vth/Vp)**2 /2) / np.sqrt(1 + (fth/ionf*Vth/Vp)**2 * (1 + 1/tau_gm) / 2E0)
    ax2.plot(fth, VB_med, color='green', label='Vlasov‒Maxwell', lw=1.5)
    fth_high = fth[fth/ionf*Vth/Vp > np.sqrt(10)]
    ERMHD_med = (fth_high/ionf*Vth/Vp) * tau_gm / np.sqrt((beta_gm*(1+tau_gm)+2*tau_gm)*(1+tau_gm))
    #ax2.plot(fth_high, ERMHD_med, color='blue', label='ERMHD', lw=1.5)

    # 軸・見た目は元コードに合わせる
    ax1.axvline(cf[1], c='purple', ls='--', lw=1, label=r'0.7 × spin frequency')
    ax1.axvline(cf[2], c='green',  ls='--', lw=1, label=r'5 × spin frequency')
    ax1.axvline(ionf, c='red', ls='--', lw=1, label=r'$f_{\mathrm{H}^{+}}$')
    ax2.axvline(cf[1], c='purple', ls='--', lw=1)
    ax2.axvline(cf[2], c='green',  ls='--', lw=1)
    ax2.axvline(ionf, c='red', ls='--', lw=1)
    for ax in (ax1, ax2):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.grid(True, which='both', ls='--', lw=0.5)
        ax.axvspan(cf[1], cf[2], color='gray', alpha=0.3)
        ax.set_xlim(cf[0], cf[3])
        ax.minorticks_on()
    ax1.set_ylim(1e-5, 1e4); ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]')
    ax1.set_yticks(np.logspace(np.log10(1e-5), np.log10(1e4), 10))
    ax2.set_ylim(1e-1, 1e3); ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$')

    # タイトルと凡例
    ax1.set_title(f'{np.datetime_as_string(t0, unit='s')} ‒ {np.datetime_as_string(t1, unit='s')}')
    ax1.legend(fontsize=10, loc='upper right')
    ax2.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    return fig

def plot_k_spectrum_dual(dd, t_start, dt_sec=1,
                         k_range=(1e-1, 1e2), n_bins=15, fit_range=(3.0, 30.0)):
    """
    k⊥ρi 軸の2段図（上: PSD, 下: 比+理論）。8Hz/128Hzを統合。
    dd: build_data_dict_xr() が返す辞書（cutoff_freq を含む前提）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    # ---- helpers ----
    def _keep_mask_by_cutoff(freq, cutoff):
        if cutoff is None:
            return np.ones_like(freq, dtype=bool)
        c = np.asarray(cutoff, float)
        if c.size == 1:
            return freq >= c[0]
        if c.size == 2:
            return (freq <= c[0]) | (freq >= c[1])
        if c.size == 3:
            return ((freq >= c[0]) & (freq <= c[1])) | (freq >= c[2])
        # >=4
        #return ((freq >= c[0]) & (freq <= c[1])) | ((freq >= c[2]) & (freq <= c[3]))
        return ((freq >= c[0]) & (freq <= c[3]))

    def _bin_collect_1d(k1d, y1d, edges):
        nb = len(edges) - 1
        out = [[] for _ in range(nb)]
        good = np.isfinite(k1d) & np.isfinite(y1d) & (k1d > 0) & (y1d > 0)
        if not np.any(good):
            return out
        bidx = np.digitize(k1d[good], edges) - 1
        inr = (bidx >= 0) & (bidx < nb)
        for yv, bi in zip(y1d[good][inr], bidx[inr]):
            out[bi].append(float(yv))
        return out

    def _geom_stats(vals, alpha=0.32):
        a = np.asarray(vals, float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return np.nan, np.nan, np.nan
        logp = np.log10(a)
        mu = np.nanmean(logp)
        lo, hi = np.nanpercentile(logp, [100*alpha/2, 100*(1-alpha/2)])
        gm = 10**mu
        return gm, gm - 10**lo, 10**hi - gm

    def _stats_bins(list_of_lists, alpha=0.32):
        gm, yerr_lo, yerr_hi = [], [], []
        for vals in list_of_lists:
            m, lo, hi = _geom_stats(vals, alpha=alpha)
            gm.append(m); yerr_lo.append(lo); yerr_hi.append(hi)
        return np.asarray(gm), np.vstack([yerr_lo, yerr_hi])  # (nbins,), (2,nbins)

    def _fit_powerlaw(x, y, xmin, xmax):
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & (x >= xmin) & (x <= xmax)
        if m.sum() < 3:
            return None
        xl, yl = np.log10(x[m]), np.log10(y[m])
        slope, intercept, _, _, stderr = linregress(xl, yl)
        return {"kappa": -slope, "stderr": stderr,
                "model": lambda xx: 10**(slope*np.log10(xx)+intercept)}

    # ---- 時間窓 ----
    t_end = t_start + np.timedelta64(dt_sec, 's')
    interp_time = _pick_window_time(dd, t_start, t_end)  # 既存ヘルパ
    if interp_time is None or getattr(interp_time, "size", 0) == 0:
        return None, None

    cf = dd.get('cutoff_freq', None)

    # ---- 物理量の窓内平均（参照 time に補間 → 平均）----
    try:
        vA   = dd["v_A_"].interp(time=interp_time).sel(time=slice(t_start, t_end)).mean().item()
        ionf = dd["ion_cyclo_freq"].interp(time=interp_time).sel(time=slice(t_start, t_end)).mean().item()
        Vp   = dd["V_ion_fac_perp"].interp(time=interp_time).sel(time=slice(t_start, t_end)).mean().item()
        Vth  = dd["V_th_proton"].interp(time=interp_time).sel(time=slice(t_start, t_end)).mean().item()
        Cs   = dd["C_s_proton"].interp(time=interp_time).sel(time=slice(t_start, t_end)).mean().item()
        tau  = dd["tau"].interp(time=interp_time).sel(time=slice(t_start, t_end)).values
        beta = dd["beta_i"].interp(time=interp_time).sel(time=slice(t_start, t_end)).values
        rpar = dd["r_param"].interp(time=interp_time).sel(time=slice(t_start, t_end)).values
    except ValueError:
        return None, None

    # ---- band毎の (time,freq) → 1D へ ----
    def _flatten_band(band):
        try:
            Ex = dd[f"Ex_{band}"].sel(time=slice(t_start, t_end)).values  # (t,F)
            Ey = dd[f"Ey_{band}"].sel(time=slice(t_start, t_end)).values
            Bx = dd[f"Bx_{band}"].sel(time=slice(t_start, t_end)).values
            By = dd[f"By_{band}"].sel(time=slice(t_start, t_end)).values
            f_full = dd[f"Ex_{band}"].freq.values                          # (F,)
        except Exception:
            return None
        if Ex.size == 0 or Bx.size == 0:
            return None

        # cutoff 適用を「最初に」実施し、全配列をスライス
        keep = _keep_mask_by_cutoff(f_full, cf)
        if not np.any(keep):
            return None
        f = f_full[keep]
        Ex = Ex[:, keep]; Ey = Ey[:, keep]
        Bx = Bx[:, keep]; By = By[:, keep]

        # k 行列を作成（freq 長は keep 後と一致）
        k_row = (f / ionf) * (Vth / Vp)                 # (Fkeep,)
        k_mat = np.tile(k_row[None, :], (Ex.shape[0], 1))  # (t,Fkeep)

        # 量を計算して 1D 化（ここで keep を二度使わない）
        Bp2    = (Bx + By) * 1e-18                      # (t,Fkeep)
        Ep2    = (Ex + Ey)                               # (t,Fkeep)
        VA2Bp2 = (vA**2) * Bp2 * 1e6
        Ratio  = np.sqrt(Ep2 * 1e-6 / Bp2) / vA

        return (k_mat.ravel(), VA2Bp2.ravel(), Ep2.ravel(), Ratio.ravel())

    collected = [tmp for band in ("8", "128") if (tmp := _flatten_band(band)) is not None]
    if not collected:
        return None, None

    k1d  = np.concatenate([c[0] for c in collected])
    v1d  = np.concatenate([c[1] for c in collected])
    e1d  = np.concatenate([c[2] for c in collected])
    r1d  = np.concatenate([c[3] for c in collected])

    # ---- ビン詰め ----
    kmin, kmax = k_range
    edges   = np.logspace(np.log10(kmin), np.log10(kmax), n_bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    xerr    = np.vstack([centers-edges[:-1], edges[1:]-centers])

    vals_V = _bin_collect_1d(k1d, v1d, edges)
    vals_E = _bin_collect_1d(k1d, e1d, edges)
    vals_R = _bin_collect_1d(k1d, r1d, edges)

    gm_V, yerr_V = _stats_bins(vals_V)  # (nbins,), (2,nbins)
    gm_E, yerr_E = _stats_bins(vals_E)
    gm_R, yerr_R = _stats_bins(vals_R)

    # ---- フィット ----
    res_B = _fit_powerlaw(centers, gm_V, fit_range[0], fit_range[1])
    res_E = _fit_powerlaw(centers, gm_E, fit_range[0], fit_range[1])

    # ---- 理論線（V–M, ERMHD）----
    tau_gm  = np.nanmedian(tau); beta_gm = np.nanmedian(beta)
    k_th = np.logspace(np.log10(kmin), np.log10(kmax), 1000)
    VB_med = (1 + (k_th**2)/2) / np.sqrt(1 + (k_th**2)*(1 + 1/tau_gm) / 2E0)
    k_th_high = k_th[k_th > np.sqrt(10)]
    ERMHD_med = k_th_high * tau_gm / np.sqrt((beta_gm*(1+tau_gm)+2*tau_gm)*(1+tau_gm))

    def _mask_err(xc, ym, yerr, xerr):
        # yerr: shape (2, nbins)
        ok = np.isfinite(ym)
        if yerr is not None:
            yerr = np.asarray(yerr, float)
            # 負値→0、非有限→NaN 扱いにして列全体を落とす
            yerr = np.where(np.isfinite(yerr), np.maximum(yerr, 0.0), np.nan)
            ok &= np.all(np.isfinite(yerr), axis=0)
        if xerr is not None:
            xerr = np.asarray(xerr, float)
            xerr = np.where(np.isfinite(xerr), np.maximum(xerr, 0.0), np.nan)
            ok &= np.all(np.isfinite(xerr), axis=0)
        return ok, (None if yerr is None else yerr[:, ok]), (None if xerr is None else xerr[:, ok]), xc[ok], ym[ok]

    # ---- 図 ----
    plt.rcdefaults(); plt.rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ok, yV, xV, cV, gV = _mask_err(centers, gm_V, yerr_V, xerr)
    okE, yE, xE, cE, gE = _mask_err(centers, gm_E, yerr_E, xerr)
    if gV.size:
        ax1.errorbar(cV, gV, yerr=yV, xerr=xV, color='b',      fmt='o', ms=4, label=r'$v_{\mathrm{A}}^2B_\perp^2$ PSD')
    if gE.size:
        ax1.errorbar(cE, gE, yerr=yE, xerr=xE, color='orange', fmt='o', ms=4, label=r'$E_\perp^2$ PSD')
    if res_B:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 200)
        ax1.plot(xx, res_B["model"](xx), '--', color='b')
        ax1.text(0.05, 0.25, r'$\kappa_{\mathrm{B}}$'+f'={res_B["kappa"]:.2f}'+r'$\pm$'+f'{res_B["stderr"]:.2f}',
                 transform=ax1.transAxes, color='b', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    if res_E:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 200)
        ax1.plot(xx, res_E["model"](xx), '--', color='orange')
        ax1.text(0.05, 0.15, r'$\kappa_{\mathrm{E}}$'+f'={res_E["kappa"]:.2f}'+r'$\pm$'+f'{res_E["stderr"]:.2f}',
                 transform=ax1.transAxes, color='orange', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]')
    ax1.set_title(f'{np.datetime_as_string(t_start, unit='s')} ‒ {np.datetime_as_string(t_end, unit='s')}')
    ax1.set_ylim(1e-5, 1e4)

    okR, yR, xR, cR, gR = _mask_err(centers, gm_R, yerr_R, xerr)
    if gR.size:
        ax2.errorbar(cR, gR, yerr=yR, xerr=xR, color='k', fmt='o', ms=4, label=r'$\sqrt{E_\perp^2/B_\perp^2}/v_{\mathrm{A}}$')
    ax2.plot(k_th,      VB_med,    color='green', lw=1.5, label='Vlasov‒Maxwell')
    #ax2.plot(k_th_high, ERMHD_med, color='blue',  lw=1.5, label='dispersion (ERMHD)')
    ax2.set_xlabel(r'$k_\perp \rho_{\mathrm{i}}$'); ax2.set_ylabel(r'$\sqrt{E_\perp^2/B_\perp^2}/v_{\mathrm{A}}$')
    ax2.set_ylim(1e-1, 1e3)

    ax1.axvline((cf[1]/ ionf) * (Vth / Vp), c='purple', ls='--', lw=1)
    ax1.axvline((cf[2] / ionf) * (Vth / Vp), c='green',  ls='--', lw=1)
    ax1.axvline(Vth / Vp, c='red', ls='--', lw=1)
    ax2.axvline((cf[1]/ ionf) * (Vth / Vp), c='purple', ls='--', lw=1, label=r'0.7 × spin frequency')
    ax2.axvline((cf[2] / ionf) * (Vth / Vp), c='green',  ls='--', lw=1, label=r'5 × spin frequency')
    ax2.axvline(Vth / Vp, c='red', ls='--', lw=1, label=r'$f_{\mathrm{H}^{+}} / f_{\mathrm{i}} \, v_{\mathrm{thi}} / V_{\mathrm{sys}\perp}$')

    for ax in (ax1, ax2):
        ax.minorticks_on()
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='both', ls='--', lw=0.5)
        ax.axvspan((cf[1]/ ionf) * (Vth / Vp), (cf[2] / ionf) * (Vth / Vp), color='gray', alpha=0.3)
        ax.set_xlim(kmin, kmax)
    ax1.set_yticks(np.logspace(np.log10(1e-5), np.log10(1e4), 10))

    ax1.legend(fontsize=10, loc='upper right')
    ax2.legend(fontsize=10, loc='upper left')
    plt.tight_layout()


    fit_results = {
        "time": t_start,
        "kappa_B": res_B["kappa"] if res_B else np.nan,
        "kappa_B_err": res_B["stderr"] if res_B else np.nan,
        "kappa_E": res_E["kappa"] if res_E else np.nan,
        "kappa_E_err": res_E["stderr"] if res_E else np.nan,
    }
    return fig, fit_results
