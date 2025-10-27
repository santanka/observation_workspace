import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ===== 1) xarray辞書を構築（ERG 64Hz専用） =====
def build_data_dict_xr_erg(ds_cwt: xr.Dataset,
                           ds_velocity_ms: xr.Dataset,
                           ds_parameter: xr.Dataset,
                           cutoff_freq=None):
    """
    ds_cwt: E/BのCWTクリーン (time,freq) を持つDataset
            例: E64_fac_x_cwt_clean, E64_fac_y_cwt_clean, B64_fac_x_cwt_clean, B64_fac_y_cwt_clean
    ds_velocity_ms: Alfven_speed_* , thermal/acoustic/perp_sys_speed (time)
    ds_parameter: proton_cycl_freq_Hz, ion_plasma_beta_* など (time)
    """
    dd = {}

    # PSD（x,yのみ使用）
    for comp in ("x","y"):
        dd[f"E{comp}"] = ds_cwt[f"E64_fac_{comp}_cwt_clean"]
        dd[f"B{comp}"] = ds_cwt[f"B64_fac_{comp}_cwt_clean"]
    dd["freq"] = ds_cwt["freq"]

    # 物理量（time基準を一意化→相互補間に使う）
    t_ref = ds_velocity_ms.time
    par_u = ds_parameter.sortby("time")
    vel_u = ds_velocity_ms.sortby("time")

    # vA（3系統）
    for tag, src in (("mid","MID"), ("hfa","HFA"), ("lepe","LEP")):
        dd[f"v_A_{tag}"] = vel_u[f"Alfven_speed_{src}"].interp(time=t_ref)

    # beta_i（3系統）
    for tag, src in (("mid","MID"), ("hfa","HFA"), ("lepe","LEP")):
        dd[f"beta_i_{tag}"] = ds_parameter[f"ion_plasma_beta_{src}"].interp(time=t_ref)

    # その他
    dd["ion_cyclo_freq"] = par_u["proton_cycl_freq_Hz"].interp(time=t_ref)
    dd["V_sys_perp"]     = vel_u["perp_sys_speed"]
    dd["V_th_ion"]       = vel_u["ion_thermal_speed"]
    dd["C_s_ion"]        = vel_u["ion_acoustic_speed"]
    dd["tau"]            = par_u['i-e_temp_ratio']
    dd['ion_mass']       = par_u['ion_mass_kg'] / 1.6726219e-27
    dd["cutoff_freq"]    = None if cutoff_freq is None else np.asarray(cutoff_freq, float)
    return dd

# ===== 2) 周波数スペクトル（MID主系列 + HFA/LEP帯） =====
def plot_freq_spectrum_erg(dd, t0, dt_sec=60, variant="mid"):
    if isinstance(dt_sec, (float, np.floating)):
        t1 = t0 + np.timedelta64(int(dt_sec * 1e6), "us")
    else:
        t1 = t0 + np.timedelta64(int(dt_sec), "s")
    f  = dd["freq"].values

    # 時間窓で平均
    Ex = dd["Ex"].sel(time=slice(t0,t1)).values if (Ex:=dd.get("Ex")) is not None else dd["Ex"]
    Ey = dd["Ey"].sel(time=slice(t0,t1)).values
    Bx = dd["Bx"].sel(time=slice(t0,t1)).values
    By = dd["By"].sel(time=slice(t0,t1)).values
    if Ex.size == 0: return None

    interp_time = dd['Ex'].time.sel(time=slice(t0,t1))

    try:
        ionf = dd["ion_cyclo_freq"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Vp   = dd["V_sys_perp"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Vth  = dd["V_th_ion"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Cs   = dd["C_s_ion"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        tau  = dd["tau"].interp(time=interp_time).sel(time=slice(t0, t1)).values
        ionmass = dd["ion_mass"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
    except ValueError:
        plt.close(fig)
        return None

    cf = dd.get('cutoff_freq')

    mEx, sEx = np.nanmean(Ex,0), np.nanstd(Ex,0)
    mEy, sEy = np.nanmean(Ey,0), np.nanstd(Ey,0)
    mBx, sBx = np.nanmean(Bx,0), np.nanstd(Bx,0)
    mBy, sBy = np.nanmean(By,0), np.nanstd(By,0)

    Bp2    = (mBx + mBy) * 1e-18         # nT^2/Hz → T^2/Hz
    Bp2err = np.sqrt(sBx**2 + sBy**2) * 1e-18
    Ep2    = (mEx + mEy)                  # (mV/m)^2/Hz
    Ep2err = np.sqrt(sEx**2 + sEy**2)

    # vA 系
    vA_main  = dd[f"v_A_{variant}"].interp(time=interp_time).sel(time=slice(t0,t1)).mean().item()
    vA_hfa   = dd["v_A_hfa"].interp(time=interp_time).sel(time=slice(t0,t1)).mean().item()
    vA_lepe  = dd["v_A_lepe"].interp(time=interp_time).sel(time=slice(t0,t1)).mean().item()

    beta_i_main  = dd[f"beta_i_{variant}"].interp(time=interp_time).sel(time=slice(t0,t1)).values
    beta_i_hfa   = dd["beta_i_hfa"].interp(time=interp_time).sel(time=slice(t0,t1)).values
    beta_i_lepe  = dd["beta_i_lepe"].interp(time=interp_time).sel(time=slice(t0,t1)).values


    # 主系列
    VA2Bp2_main = (vA_main**2) * Bp2 * 1e6
    ratio_main  = np.sqrt(Ep2*1e-6 / Bp2) / vA_main
    ratio_err   = ratio_main * np.sqrt(0.25*(Ep2err/Ep2)**2 + 0.25*(Bp2err/Bp2)**2)

    # 帯域（上限=LEP, 下限=HFA）
    VA2_hfa  = (vA_hfa**2)  * Bp2 * 1e6
    VA2_lepe = (vA_lepe**2) * Bp2 * 1e6
    Q = np.sqrt(Ep2*1e-6 / Bp2)
    r_lo, r_hi = Q/vA_hfa, Q/vA_lepe

    # マスク（任意のcutoffをNaN化）
    def _mask(arr):
        c = dd["cutoff_freq"]
        if c is None: return arr
        mask = np.ones_like(f, bool)
        if c.size==2: mask = (f<=c[0]) | (f>=c[1])
        elif c.size==3: mask = ((f>=c[0])&(f<=c[1])) | (f>=c[2])
        elif c.size>=4: mask = ((f>=c[0])&(f<=c[3]))
        #elif c.size>=4: mask = ((f>=c[0])&(f<=c[1])) | ((f>=c[2])&(f<=c[3]))
        out = np.array(arr,copy=True); out[~mask]=np.nan; return out

    Ep2_main = _mask(Ep2); Ep2err=_mask(Ep2err); Bp2err=_mask(Bp2err)
    VA2Bp2_main = _mask(VA2Bp2_main); VA2_hfa=_mask(VA2_hfa); VA2_lepe=_mask(VA2_lepe)
    ratio_main  = _mask(ratio_main);  ratio_err=_mask(ratio_err); r_lo=_mask(r_lo); r_hi=_mask(r_hi)

    # 図
    plt.rcdefaults()
    plt.rcParams['font.size'] = 12
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,7),sharex=True)

    # 上段：PSD
    ax1.fill_between(f, np.minimum(VA2_hfa,VA2_lepe), np.maximum(VA2_hfa,VA2_lepe),
                     alpha=0.6, label=r'$v_{\mathrm{A}}^2B_\perp^2$ [HFA–LEP]')
    ax1.errorbar(f, VA2Bp2_main, yerr=(vA_main**2)*Bp2err*1e6, fmt='o', ms=3, color='b',
                 label=r'$v_{\mathrm{A}}^2B_\perp^2$' + f' [{variant}]')
    ax1.errorbar(f, Ep2_main, yerr=Ep2err, fmt='o', ms=3, color='orange', label=r'$E_\perp^2$')
    ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]'); ax1.set_ylim(1e-5,1e4); ax1.set_xlim(1e-2,32)

    # 下段：比
    fth = np.logspace(-2, np.log10(32), 1000)
    tau_gm  = np.nanmedian(tau)
    VB_med = (1 + (fth/ionf*ionmass*Vth/Vp)**2 /2) / np.sqrt(1 + (fth/ionf*ionmass*Vth/Vp)**2 * (1 + 1/tau_gm) / 2E0)
    ax2.plot(fth, VB_med, color='green', label='Vlasov‒Maxwell', lw=1.5)

    ax2.fill_between(f, np.minimum(r_lo,r_hi), np.maximum(r_lo,r_hi), alpha=0.6,
                     label=r'$\sqrt{E_\perp^2/B_\perp^2}/v_{\mathrm{A}}$ [HFA–LEP]')
    ax2.errorbar(f, ratio_main, yerr=ratio_err, fmt='o', ms=3, color='k',
                 label=r'$\sqrt{{E_\perp^2/B_\perp^2}}/v_{\mathrm{A}}$' + f' [{variant}]')
    ax2.set_xlabel('Frequency [Hz]'); ax2.set_ylabel(r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$')
    ax2.set_ylim(1e-1,1e3)

    ax1.axvline(cf[1], c='purple', ls='--', lw=1, label=r'0.7 × spin tone')
    ax1.axvline(cf[2], c='green',  ls='--', lw=1, label=r'5 × spin tone')
    ax1.axvline(ionf, c='red', ls='--', lw=1, label=r'$f_{\mathrm{H}^{+}}$')
    ax2.axvline(cf[1], c='purple', ls='--', lw=1)
    ax2.axvline(cf[2], c='green',  ls='--', lw=1)
    ax2.axvline(ionf, c='red', ls='--', lw=1)
    for ax in (ax1, ax2):
        ax.minorticks_on()
        ax.grid(True, which='both', ls='--', lw=0.5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.axvspan(cf[1], cf[2], color='gray', alpha=0.3)
        ax.set_xlim(1e-2, 32)
    ax1.set_yticks(np.logspace(np.log10(1e-5), np.log10(1e4), 10))
    
    if isinstance(dt_sec, (float, np.floating)):
        ax1.set_title(f'{np.datetime_as_string(t0, unit="us")} ‒ {np.datetime_as_string(t1, unit="us")}')
    else:
        ax1.set_title(f'{np.datetime_as_string(t0, unit='s')} ‒ {np.datetime_as_string(t1, unit='s')}')
    ax1.legend(fontsize=10, loc='lower left', ncols=2)
    ax2.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    return fig

# ===== 3) k⊥ρi 図（1秒ビン。64Hzのみをflatten→ビン詰め） =====
def plot_k_spectrum_erg(dd, t0, dt_sec=1, k_range=(1e-1,1e2), n_bins=15, fit_range=(3,30), variant="mid"):
    from scipy.stats import linregress

    if isinstance(dt_sec, (float, np.floating)):
        t1 = t0 + np.timedelta64(int(dt_sec * 1e6), "us")
    else:
        t1 = t0 + np.timedelta64(int(dt_sec), "s")
    f  = dd["freq"].values

    # 時間窓で平均
    Ex = dd["Ex"].sel(time=slice(t0,t1)).values if (Ex:=dd.get("Ex")) is not None else dd["Ex"]
    Ey = dd["Ey"].sel(time=slice(t0,t1)).values
    Bx = dd["Bx"].sel(time=slice(t0,t1)).values
    By = dd["By"].sel(time=slice(t0,t1)).values
    if Ex.size == 0: return None

    interp_time = dd['Ex'].time.sel(time=slice(t0,t1))
    try:
        ionf = dd["ion_cyclo_freq"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Vp   = dd["V_sys_perp"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Vth  = dd["V_th_ion"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        Cs   = dd["C_s_ion"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        tau  = dd["tau"].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
        ionmass = dd['ion_mass'].interp(time=interp_time).sel(time=slice(t0, t1)).mean().item()
    except ValueError:
        plt.close(fig)
        return None

    # vA 系
    vA_main  = dd[f"v_A_{variant}"].interp(time=interp_time).sel(time=slice(t0,t1)).mean().item()
    vA_hfa   = dd["v_A_hfa"].interp(time=interp_time).sel(time=slice(t0,t1)).mean().item()
    vA_lepe  = dd["v_A_lepe"].interp(time=interp_time).sel(time=slice(t0,t1)).mean().item()

    beta_i_main  = dd[f"beta_i_{variant}"].interp(time=interp_time).sel(time=slice(t0,t1)).values
    beta_i_hfa   = dd["beta_i_hfa"].interp(time=interp_time).sel(time=slice(t0,t1)).values
    beta_i_lepe  = dd["beta_i_lepe"].interp(time=interp_time).sel(time=slice(t0,t1)).values

    cf = dd.get('cutoff_freq')
    f  = dd["freq"].values

    def _freq_mask(f, cf):
        if cf is None: 
            return np.ones_like(f, dtype=bool)
        m = np.ones_like(f, dtype=bool)
        if cf.size == 2:
            m = (f <= cf[0]) | (f >= cf[1])
        elif cf.size == 3:
            m = ((f >= cf[0]) & (f <= cf[1])) | (f >= cf[2])
        elif cf.size >= 4:
            m = ((f >= cf[0]) & (f <= cf[3]))
            #m = ((f >= cf[0]) & (f <= cf[1])) | ((f >= cf[2]) & (f <= cf[3]))
        return m

    m_f = _freq_mask(f, cf)                         # (F,)
    m_2d = np.broadcast_to(m_f, Ex.shape)          # (T,F)

    # freqカットを行列に適用
    Ex = np.where(m_2d, Ex, np.nan)
    Ey = np.where(m_2d, Ey, np.nan)
    Bx = np.where(m_2d, Bx, np.nan)
    By = np.where(m_2d, By, np.nan)

    # k行列と量（主系列は variant の vA、帯は hfa/lepe）
    k_row = (f/ionf*ionmass) * (Vth/Vp)                     # (F,)
    k_mat = np.tile(k_row[None, :], (Ex.shape[0], 1))
    Bp2   = (Bx + By) * 1e-18                       # → T^2/Hz
    Ep2   = (Ex + Ey)                               # → (mV/m)^2/Hz
    vA2_main = vA_main**2

    # 主系列の行列量
    VA2Bp2_main_mat = vA2_main * Bp2 * 1e6          # (mV/m)^2/Hz
    with np.errstate(divide='ignore', invalid='ignore'):
        Ratio_main_mat = np.sqrt(Ep2 * 1e-6 / Bp2) / vA_main

    # --- ビン詰め（幾何平均＋分位幅） ---
    edges   = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_bins+1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    xerr    = np.vstack([centers-edges[:-1], edges[1:]-centers])

    def _bin(arr_k, arr_y, edges):
        nb=len(edges)-1; res=[[] for _ in range(nb)]
        m = np.isfinite(arr_k)&np.isfinite(arr_y)&(arr_k>0)&(arr_y>0)
        b = np.digitize(arr_k[m], edges)-1
        ok=(b>=0)&(b<nb)
        for y,bi in zip(arr_y[m][ok], b[ok]): res[bi].append(float(y))
        return res

    def _gstats(listvals, alpha=0.32):
        gm,lo,hi=[],[],[]
        for vals in listvals:
            a=np.asarray(vals,float); a=a[np.isfinite(a)&(a>0)]
            if a.size==0: gm.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            lg=np.log10(a); mu=np.nanmean(lg)
            ql,qh=np.nanpercentile(lg,[100*alpha/2,100*(1-alpha/2)])
            g=10**mu; gm.append(g); lo.append(g-10**ql); hi.append(10**qh-g)
        return np.asarray(gm), np.vstack([lo,hi])

    # 主系列（variant）のビン統計
    gm_VA2, yV = _gstats(_bin(k_mat.ravel(), VA2Bp2_main_mat.ravel(), edges))
    gm_E,   yE = _gstats(_bin(k_mat.ravel(), Ep2.ravel(),              edges))
    gm_R,   yR = _gstats(_bin(k_mat.ravel(), Ratio_main_mat.ravel(),   edges))

    # --- hfa/lepe 帯の構築（freq版と同じ仕様） ---
    # まず B⊥^2 と √(E⊥^2/B⊥^2) を k ビンで幾何平均
    gm_Bperp2, _ = _gstats(_bin(k_mat.ravel(), (Bx+By).ravel(), edges))       # nT^2/Hz
    gm_Q,      _ = _gstats(_bin(k_mat.ravel(), np.sqrt(Ep2*1e-6/Bp2).ravel(), edges))  # √(E/B)

    def _fit_powerlaw_xy(x, y, xmin, xmax):
        m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0) & (x>=xmin) & (x<=xmax)
        if m.sum() < 3: return None
        xl, yl = np.log10(x[m]), np.log10(y[m])
        s, i, _, _, se = linregress(xl, yl)
        return {"kappa": -s, "stderr": se, "model": lambda xx: 10**(s*np.log10(xx)+i)}
    
    res_B = _fit_powerlaw_xy(centers, gm_VA2, fit_range[0], fit_range[1])
    res_E = _fit_powerlaw_xy(centers, gm_E,   fit_range[0], fit_range[1])

    # 1秒内平均 vA（hfa, lepe）
    m_v_hfa  = vA_hfa
    m_v_lepe = vA_lepe

    # PSD 側の帯：vA^2 B⊥^2（上限=lepe, 下限=hfa）。単位変換を合わせる
    VA2B2_hfa_bins  = (m_v_hfa**2)  * (gm_Bperp2 * 1e-18) * 1e6
    VA2B2_lepe_bins = (m_v_lepe**2) * (gm_Bperp2 * 1e-18) * 1e6
    valid_psd = np.isfinite(VA2B2_hfa_bins) & np.isfinite(VA2B2_lepe_bins) & (VA2B2_hfa_bins>0) & (VA2B2_lepe_bins>0)
    lower_psd = np.minimum(VA2B2_hfa_bins,  VA2B2_lepe_bins)
    upper_psd = np.maximum(VA2B2_hfa_bins,  VA2B2_lepe_bins)

    # 比の帯：√(E⊥^2/B⊥^2)/vA（上限=lepe, 下限=hfa 固定）
    ratio_hfa_bins  = gm_Q / m_v_hfa
    ratio_lepe_bins = gm_Q / m_v_lepe
    valid_ratio = np.isfinite(ratio_hfa_bins) & np.isfinite(ratio_lepe_bins) & (ratio_hfa_bins>0) & (ratio_lepe_bins>0)
    lower_r = np.minimum(ratio_hfa_bins, ratio_lepe_bins)
    upper_r = np.maximum(ratio_hfa_bins, ratio_lepe_bins)

    # 理論
    kth = np.logspace(-1, 2, 1000)
    VB_th = (1 + kth**2 / 2) / np.sqrt(1 + kth**2 * (1 + 1/tau) / 2E0)


    # --- 描画（主系列=variant を点＋誤差、帯を背面に） ---
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,7),sharex=True)
    for ax in (ax1,ax2): ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True,ls='--',lw=0.5, which='both')

    #if cf is not None:
    #    cf = np.asarray(cf, float)
    #    # 窓内代表で f→k
    #    def f2k(fc): return (fc/ionf)*(Vth/Vp)
#
    #    if cf.size >= 2:
    #        k1, k2 = f2k(cf[1]), f2k(cf[2])
    #        lo, hi = (min(k1, k2), max(k1, k2))
    #        for ax in (ax1, ax2):
    #            ax.axvline(lo, ls='--', lw=1, color='purple')
    #            ax.axvline(hi, ls='--', lw=1, color='green')
    #            ax.axvspan(lo, hi, color='gray', alpha=0.3)

    # PSD：帯
    ax1.fill_between(centers, lower_psd, upper_psd, where=valid_psd, alpha=0.6,
                     label=r'$v_{\mathrm{A}}^2B_\perp^2$ [HFA–LEP]')
    # 主系列
    ax1.errorbar(centers, gm_VA2, yerr=np.clip(yV,0,np.inf), xerr=xerr, fmt='o', ms=4, color='b',
                 label=r'$v_{\mathrm{A}}^2B_\perp^2$' + f' [{variant}]')
    ax1.errorbar(centers, gm_E,   yerr=np.clip(yE,0,np.inf), xerr=xerr, fmt='o', ms=4, color='orange',
                 label=r'$E_\perp^2$')
    
    if res_B:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 200)
        ax1.plot(xx, res_B["model"](xx), '--', color='b')
        ax1.text(0.05, 0.25,
                 r'$\kappa_{\mathrm{B}}$' + f'={res_B["kappa"]:.2f}' + r'$\pm$' + f'{res_B["stderr"]:.2f}',
                 transform=ax1.transAxes, color='b',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    if res_E:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 200)
        ax1.plot(xx, res_E["model"](xx), '--', color='orange')
        ax1.text(0.05, 0.15,
                 r'$\kappa_{\mathrm{E}}$' + f'={res_E["kappa"]:.2f}' + r'$\pm$' + f'{res_E["stderr"]:.2f}',
                 transform=ax1.transAxes, color='orange',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]'); ax1.set_ylim(1e-5,1e4); ax1.set_xlim(*k_range)
    if isinstance(dt_sec, (float, np.floating)):
        ax1.set_title(f'{np.datetime_as_string(t0, unit="us")} ‒ {np.datetime_as_string(t1, unit="us")}')
    else:
        ax1.set_title(f'{np.datetime_as_string(t0, unit='s')} ‒ {np.datetime_as_string(t1, unit='s')}')

    # 比：帯
    ax2.plot(kth, VB_th, color='green', label='Vlasov‒Maxwell', lw=1.5)
    ax2.fill_between(centers, lower_r, upper_r, where=valid_ratio, alpha=0.6,
                     label=r'$\sqrt{E_\perp^2/B_\perp^2}/v_{\mathrm{A}}$ [HFA–LEP]')
    # 主系列
    ax2.errorbar(centers, gm_R, yerr=np.clip(yR,0,np.inf), xerr=xerr, fmt='o', ms=4, color='k',
                 label=r'$\sqrt{{E_\perp^2/B_\perp^2}}/v_{\mathrm{A}}$' + f' [{variant}]')
    ax2.set_xlabel(r'$k_\perp\rho_{\mathrm{i}}$'); ax2.set_ylabel(r'$\sqrt{E_\perp^2/B_\perp^2}/v_{\mathrm{A}}$')
    ax2.set_ylim(1e-1,1e3)

    ax1.axvline((cf[1]/ ionf) * (Vth / Vp), c='purple', ls='--', lw=1)
    ax1.axvline((cf[2] / ionf) * (Vth / Vp), c='green',  ls='--', lw=1)
    ax1.axvline(Vth / Vp * ionmass, c='red', ls='--', lw=1)
    ax2.axvline((cf[1]/ ionf) * (Vth / Vp), c='purple', ls='--', lw=1, label=r'0.7 × spin frequency')
    ax2.axvline((cf[2] / ionf) * (Vth / Vp), c='green',  ls='--', lw=1, label=r'5 × spin frequency')
    ax2.axvline(Vth / Vp * ionmass, c='red', ls='--', lw=1, label=r'$f_{\mathrm{H}^{+}} / f_{\mathrm{i}} \, v_{\mathrm{thi}} / V_{\mathrm{sys}\perp}$')

    for ax in (ax1, ax2):
        ax.minorticks_on()
        ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='both', ls='--', lw=0.5)
        ax.axvspan((cf[1]/ ionf) * (Vth / Vp), (cf[2] / ionf) * (Vth / Vp), color='gray', alpha=0.3)
        ax.set_xlim(k_range[0], k_range[1])
    
    ax1.set_yticks(np.logspace(np.log10(1e-5), np.log10(1e4), 10))

    ax1.legend(fontsize=10, loc='upper right')
    ax2.legend(fontsize=10, loc='upper left', ncols=2)
    plt.tight_layout()

    fit_results = {
        "time": t0,
        "kappa_B": res_B["kappa"] if res_B else np.nan,
        "kappa_B_err": res_B["stderr"] if res_B else np.nan,
        "kappa_E": res_E["kappa"] if res_E else np.nan,
        "kappa_E_err": res_E["stderr"] if res_E else np.nan,
    }
    return fig, fit_results