# psd_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import pytplot as pt
import xarray as xr
from scipy.stats import chi2, linregress
from numpy.linalg import LinAlgError

# -----------------------------------------------------------
# データ読み込み/準備
# -----------------------------------------------------------
def load_and_prepare_data(cutoff_freq=1/6):
    """
    pytplotから必要なデータを読み込み、前処理を行う。
    戻り値: データと周波数軸を含む辞書
    """
    data_dict = {}
    try:
        # --- 生の時系列データを取得 ---
        data_dict['Bx_psd'] = pt.data_quants['B_fac_x_BS_clean']
        data_dict['By_psd'] = pt.data_quants['B_fac_y_BS_clean']
        data_dict['Ex_psd'] = pt.data_quants['E_fac_x_BS_clean']
        data_dict['Ey_psd'] = pt.data_quants['E_fac_y_BS_clean']
        
        time_axis = data_dict['Bx_psd'].time.values # 基準となる時間軸
        data_dict['time'] = time_axis
        
        # --- 補助パラメータを基準時間軸に補間 ---
        data_dict['v_A_'] = pt.data_quants['Alfven_speed'].interp(time=time_axis, method='linear') * 1E3
        data_dict['ion_cyclo_freq'] = pt.data_quants['ion_cyclo_freq'].interp(time=time_axis, method='linear')
        data_dict['V_ion_fac_perp'] = pt.data_quants['v_sys_fac_perp_rolling'].interp(time=time_axis, method='linear') * 1E3
        data_dict['V_th_proton'] = pt.data_quants['V_th_proton'].interp(time=time_axis, method='linear') * 1E3
        data_dict['C_s_proton'] = pt.data_quants['C_s_proton'].interp(time=time_axis, method='linear') * 1E3
        data_dict['tau'] = pt.data_quants['tau'].interp(time=time_axis, method='linear')
        data_dict['beta_i'] = pt.data_quants['beta_i'].interp(time=time_axis, method='linear')
        data_dict['r_param'] = data_dict['C_s_proton'] / data_dict['V_th_proton']

        
        # --- 周波数軸の準備とカットオフ ---
        freq = data_dict['Bx_psd'].v.values
        if cutoff_freq is not None:
            if len(cutoff_freq) == 2:
                mask = (freq <= cutoff_freq[0]) | (freq >= cutoff_freq[1])
            elif len(cutoff_freq) == 3:
                mask = ((freq >= cutoff_freq[0]) & (freq <= cutoff_freq[1])) | (freq >= cutoff_freq[2])
            else:
                mask = freq >= cutoff_freq
            data_dict['freq'] = freq[mask]
        else:
            data_dict['freq'] = freq
        
        # 各PSDデータにマスクを適用
        for key in ['Bx_psd', 'By_psd', 'Ex_psd', 'Ey_psd']:
            if cutoff_freq is not None:
                data_dict[key] = data_dict[key].isel(v_dim=mask)
            else:
                data_dict[key] = data_dict[key]

        # --- k_perp*rho_i を全時刻分あらかじめ計算 ---
        f     = data_dict['freq']
        w_ci  = data_dict['ion_cyclo_freq'].values
        vth_i = data_dict['V_th_proton'].values
        vperp = data_dict['V_ion_fac_perp'].values
        data_dict['kperp_rhoi'] = (f[None, :] / w_ci[:, None]) * (vth_i[:, None] / vperp[:, None])

    except Exception as e:
        print(f"データの読み込みに失敗しました: {e}")
        return None
        
    return data_dict

# -----------------------------------------------------------
# kモードプロット用の補助関数
# -----------------------------------------------------------
def bin_fill(k_mat, y_mat, edges):
    nb = len(edges) - 1
    res = [[] for _ in range(nb)]
    for k_row, y_row in zip(k_mat, y_mat):
        good = np.isfinite(k_row) & np.isfinite(y_row) & (k_row > 0)
        if not np.any(good): continue
        bidx = np.digitize(k_row[good], edges) - 1
        in_range = (bidx >= 0) & (bidx < nb)
        for y_i, bi in zip(y_row[good][in_range], bidx[in_range]):
            res[bi].append(y_i)
    return res

def geom_stats(values, mode='quantile', alpha=0.32):
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0: return np.nan, np.nan, np.nan
    logp = np.log10(arr)
    mu = logp.mean()
    if mode == 'quantile':
        lo, hi = np.nanpercentile(logp, [100*alpha/2, 100*(1-alpha/2)])
    else: raise ValueError("mode must be 'quantile'.")
    gm = 10**mu
    return gm, gm - 10**lo, 10**hi - gm

def stats_bins(list_of_lists, mode='quantile', alpha=0.32):
    gm, lo, hi = [], [], []
    for vals in list_of_lists:
        m, l, h = geom_stats(vals, mode=mode, alpha=alpha)
        gm.append(m); lo.append(l); hi.append(h)
    return np.array(gm), np.vstack([lo, hi])

def fit_powerlaw_loglog(x, y, *, yerr=None, mask=None):
    if mask is None: mask = np.ones_like(x, dtype=bool)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    use = valid & mask
    if use.sum() < 3: return None
    Xl, Yl = np.log10(x[use]), np.log10(y[use])
    try:
        slope, intercept, *_ = linregress(Xl, Yl)
        stderr = np.nan # linregressのstderrは使わない
    except (LinAlgError, ValueError): return None
    return {"kappa": -slope, "stderr": stderr, "model": lambda xx: 10**(slope*np.log10(xx)+intercept)}

def draw_lognormal(gm, lo, hi, size, rng):
    mu_log = np.log10(gm)
    sig_hi = np.log10(gm + hi) - mu_log if np.isfinite(hi) else 0.0
    sig_lo = mu_log - np.log10(gm - lo) if np.isfinite(lo) and (gm - lo) > 0 else sig_hi
    sigma  = np.nanmean([sig_hi, sig_lo])
    return 10**rng.normal(mu_log, sigma, size=size)

def ci_band(arr2d):
    return np.nanpercentile(arr2d, [16, 50, 84], axis=0)

# -----------------------------------------------------------
# 新しいプロット関数: kモード
# -----------------------------------------------------------
def plot_k_spectrum(t_start, data_dict, interval_sec=1, n_samples_mc=500,
                    k_range=(1e-1, 1e2), n_bins=15, 
                    fit_range=(3.0, 30.0)):
    """
    指定された時刻から1秒間のデータをk空間でビン詰めし、
    PSDとE/B比の2段組プロットを作成する。
    """
    t_end = t_start + np.timedelta64(interval_sec, 's')
    time_axis = data_dict['time']
    
    idx_t = (time_axis >= t_start) & (time_axis < t_end)
    if idx_t.sum() < 2:
        print(f"{t_start} のデータが2点未満です。スキップします。")
        return None

    # --- 1秒間のデータをスライス ---
    Bx_sel = data_dict['Bx_psd'].values[idx_t, :]
    By_sel = data_dict['By_psd'].values[idx_t, :]
    Ex_sel = data_dict['Ex_psd'].values[idx_t, :]
    Ey_sel = data_dict['Ey_psd'].values[idx_t, :]
    k_sel = data_dict['kperp_rhoi'][idx_t, :]
    vA2_sel = data_dict['v_A_'].values[idx_t]**2
    
    # 派生物理量を計算
    VA2_Bperp2_sel = vA2_sel[:, None] * (Bx_sel + By_sel) * 1e-12
    E_perp2_sel = Ex_sel + Ey_sel
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_sel = np.sqrt(E_perp2_sel / VA2_Bperp2_sel)

    # --- ビン詰めと統計計算 ---
    kmin, kmax = k_range
    bin_edges = np.logspace(np.log10(kmin), np.log10(kmax), n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    xerr = np.vstack([bin_centers - bin_edges[:-1], bin_edges[1:] - bin_centers])

    vals_VA2_bins = bin_fill(k_sel, VA2_Bperp2_sel, bin_edges)
    vals_E_bins = bin_fill(k_sel, E_perp2_sel, bin_edges)
    vals_R_bins = bin_fill(k_sel, ratio_sel, bin_edges)

    gm_VA2, yerr_VA2 = stats_bins(vals_VA2_bins)
    gm_E, yerr_E = stats_bins(vals_E_bins)
    gm_R, yerr_R = stats_bins(vals_R_bins)

    yerr_VA2, yerr_E, yerr_R, xerr = map(lambda err: np.clip(err, 0, np.inf), [yerr_VA2, yerr_E, yerr_R, xerr])

    # --- Power-lawフィッティング ---
    fit_mask = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
    res_B = fit_powerlaw_loglog(bin_centers, gm_VA2, yerr=yerr_VA2, mask=fit_mask)
    res_E = fit_powerlaw_loglog(bin_centers, gm_E, yerr=yerr_E, mask=fit_mask)

    # --- 理論曲線のためのパラメータ統計 ---
    tau_sel = data_dict['tau'][idx_t]
    beta_sel = data_dict['beta_i'][idx_t]
    r_sel = data_dict['r_param'][idx_t]
    
    gm_tau, lo_tau, hi_tau = geom_stats(tau_sel)
    gm_beta, lo_beta, hi_beta = geom_stats(beta_sel)
    gm_r, lo_r, hi_r = geom_stats(r_sel)
    
    # --- モンテカルロでエラーバンド計算 ---
    rng = np.random.default_rng()
    tau_s = draw_lognormal(gm_tau, lo_tau, hi_tau, n_samples_mc, rng)
    beta_s = draw_lognormal(gm_beta, lo_beta, hi_beta, n_samples_mc, rng)
    r_s = draw_lognormal(gm_r, lo_r, hi_r, n_samples_mc, rng)
    
    k_th = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 1000)
    VB_samp = (1 + k_th[None, :]**2) / np.sqrt(1 + k_th[None, :]**2 * (1 + r_s[:, None]**2))
    
    k_th_high = k_th[k_th > np.sqrt(10)]
    ERMHD_samp = k_th_high[None, :] * tau_s[:, None] / np.sqrt((beta_s[:, None]*(1+tau_s[:, None]) + 2*tau_s[:, None])*(1+tau_s[:, None]))

    VB_low, VB_med, VB_high = ci_band(VB_samp)
    ERMHD_low, ERMHD_med, ERMHD_high = ci_band(ERMHD_samp)
    
    # --- プロット作成 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # --- 上段: PSD ---
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.errorbar(bin_centers, gm_VA2, yerr=yerr_VA2, xerr=xerr, color='blue', fmt='o', ms=4, label=r'$v_{\mathrm{A}}^2 B_\perp^2$ PSD')
    ax1.errorbar(bin_centers, gm_E, yerr=yerr_E, xerr=xerr, color='orange', fmt='o', ms=4, label=r'$E_\perp^2$ PSD')
    if res_B:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 100)
        ax1.plot(xx, res_B["model"](xx), '--', color='blue')
        ax1.text(0.05, 0.25, fr'$\kappa_B={res_B["kappa"]:.2f}$', transform=ax1.transAxes, color='blue',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    if res_E:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 100)
        ax1.plot(xx, res_E["model"](xx), '--', color='orange')
        ax1.text(0.05, 0.15, fr'$\kappa_E={res_E["kappa"]:.2f}$', transform=ax1.transAxes, color='orange',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    ax1.set_ylabel(r'PSD [(mV/m)$^2$/Hz]')
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    ax1.set_title(f'Binned PSD @ {str(t_start)}')
    ax1.legend()
    ax1.set_xlim(kmin, kmax); ax1.set_ylim(1e-6, 1e3)

    # --- 下段: E/B比と理論曲線 ---
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.errorbar(bin_centers, gm_R, yerr=yerr_R, xerr=xerr, color='k', fmt='o', ms=4, label=r'$\sqrt{E_\perp^2/B_\perp^2} / v_A$')
    
    ax2.plot(k_th, VB_med, color='green', lw=1.5, label='dispersion relation (V-M)')
    ax2.fill_between(k_th, VB_low, VB_high, color='green', alpha=0.2)
    
    ax2.plot(np.logspace(-1, np.log10(np.sqrt(0.1)), 100), np.ones(100), color='blue', lw=1.5, label='dispersion relation (ERMHD)')
    ax2.plot(k_th_high, ERMHD_med, color='blue', lw=1.5)
    ax2.fill_between(k_th_high, ERMHD_low, ERMHD_high, color='blue', alpha=0.2)

    ax2.set_xlabel(r'$\omega_{\mathrm{sc}} \rho_{\mathrm{i}} / |V_{\mathrm{flow}\perp}|$')
    ax2.set_ylabel(r'$\sqrt{E_\perp^2/B_\perp^2} / v_A$')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.legend()
    ax2.set_xlim(kmin, kmax); ax2.set_ylim(1e-1, 1e3)
    
    plt.tight_layout()
    return fig


def plot_freq_spectrum(t_start, data_dict, interval_sec=1, n_samples_mc=500):
    """
    指定された時刻から1秒間の平均スペクトルと分散（エラーバー/バンド）を計算し、プロットを作成する。
    """
    t_end = t_start + np.timedelta64(interval_sec, 's')
    freq = data_dict['freq']

    freq_th = np.logspace(-2, 2, 1000)

    try:
        Bx_psd_sec = data_dict['Bx_psd'].sel(time=slice(t_start, t_end))
        if Bx_psd_sec.time.size < 2: # 分散を計算するには2点以上必要
            print(f"{t_start} のデータが2点未満です。スキップします。")
            return None
            
        By_psd_sec = data_dict['By_psd'].sel(time=slice(t_start, t_end))
        Ex_psd_sec = data_dict['Ex_psd'].sel(time=slice(t_start, t_end))
        Ey_psd_sec = data_dict['Ey_psd'].sel(time=slice(t_start, t_end))
    except Exception as e:
        print(f"データスライスの取得に失敗: {e}")
        return None

    # --- 時間平均と標準偏差を計算 ---
    mean_Bx = np.nanmean(Bx_psd_sec.values, axis=0)
    mean_By = np.nanmean(By_psd_sec.values, axis=0)
    std_Bx = np.nanstd(Bx_psd_sec.values, axis=0)
    std_By = np.nanstd(By_psd_sec.values, axis=0)
    
    mean_Ex = np.nanmean(Ex_psd_sec.values, axis=0)
    mean_Ey = np.nanmean(Ey_psd_sec.values, axis=0)
    std_Ex = np.nanstd(Ex_psd_sec.values, axis=0)
    std_Ey = np.nanstd(Ey_psd_sec.values, axis=0)

    # 誤差伝播（和の標準偏差は、各標準偏差の二乗和の平方根）
    B_perp2_slice = (mean_Bx + mean_By) * 1e-18
    E_perp2_slice = mean_Ex + mean_Ey
    B_perp2_err = np.sqrt(std_Bx**2 + std_By**2) * 1e-18
    E_perp2_err = np.sqrt(std_Ex**2 + std_Ey**2)

    # --- 補助パラメータも時間平均と標準偏差を計算 ---
    v_A_sec = data_dict['v_A_'].sel(time=slice(t_start, t_end))
    mean_v_A, std_v_A = v_A_sec.mean().values, v_A_sec.std().values

    ion_cyclo_sec = data_dict['ion_cyclo_freq'].sel(time=slice(t_start, t_end))
    mean_ion_cyclo, std_ion_cyclo = ion_cyclo_sec.mean().values, ion_cyclo_sec.std().values

    V_ion_sec = data_dict['V_ion_fac_perp'].sel(time=slice(t_start, t_end))
    mean_V_ion, std_V_ion = V_ion_sec.mean().values, V_ion_sec.std().values

    V_th_sec = data_dict['V_th_proton'].sel(time=slice(t_start, t_end))
    mean_V_th, std_V_th = V_th_sec.mean().values, V_th_sec.std().values

    C_s_sec = data_dict['C_s_proton'].sel(time=slice(t_start, t_end))
    mean_C_s, std_C_s = C_s_sec.mean().values, C_s_sec.std().values
    
    # 平均値を使った計算
    VA2_Bperp2_slice = (mean_v_A**2) * B_perp2_slice * 1e6
    ratio_dimless    = np.sqrt(E_perp2_slice * 1e-6 / B_perp2_slice) / mean_v_A

    # E/B比のエラーを誤差伝播で計算
    # R = sqrt(E/B)/V なので、(σR/R)^2 = (1/4)(σE/E)^2 + (1/4)(σB/B)^2 + (σV/V)^2
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err_E_sq = (E_perp2_err / E_perp2_slice)**2
        rel_err_B_sq = (B_perp2_err / B_perp2_slice)**2
        rel_err_V_sq = (std_v_A / mean_v_A)**2
        
        # 0やnanによるエラーを回避
        rel_err_E_sq = np.nan_to_num(rel_err_E_sq)
        rel_err_B_sq = np.nan_to_num(rel_err_B_sq)
        rel_err_V_sq = np.nan_to_num(rel_err_V_sq)

        total_rel_err_sq = 0.25 * rel_err_E_sq + 0.25 * rel_err_B_sq + rel_err_V_sq
        ratio_err = ratio_dimless * np.sqrt(total_rel_err_sq)

    # --- モンテカルロ法で理論曲線のエラーバンドを計算 ---
    rng = np.random.default_rng()
    v_A_s = rng.normal(mean_v_A, std_v_A, size=n_samples_mc)
    ion_cyclo_s = rng.normal(mean_ion_cyclo, std_ion_cyclo, size=n_samples_mc)
    V_ion_s = rng.normal(mean_V_ion, std_V_ion, size=n_samples_mc)
    V_th_s = rng.normal(mean_V_th, std_V_th, size=n_samples_mc)
    C_s_s = rng.normal(mean_C_s, std_C_s, size=n_samples_mc)
    
    # 理論曲線の計算に freq_th を使う
    VB_samples = (1 + (freq_th[None, :] / ion_cyclo_s[:, None] * V_th_s[:, None] / V_ion_s[:, None])**2) / \
                 np.sqrt(1 + (freq_th[None, :] / ion_cyclo_s[:, None])**2 * ((C_s_s[:, None] / V_ion_s[:, None])**2 + (V_th_s[:, None] / V_ion_s[:, None])**2))

    tau_s = (V_th_s / C_s_s)**2 / 2
    beta_i_s = (V_th_s / v_A_s)**2
    kperp_rhoi_s = (freq_th[None, :] / ion_cyclo_s[:, None]) * (V_th_s[:, None] / V_ion_s[:, None]) # freq_th を使用
    high_val_s = kperp_rhoi_s * tau_s[:, None] / np.sqrt((beta_i_s[:, None] * (1 + tau_s[:, None]) + 2 * tau_s[:, None]) * (1 + tau_s[:, None]))
    ERMHD_samples = np.select([kperp_rhoi_s < np.sqrt(0.1), kperp_rhoi_s > np.sqrt(10.0)], [np.ones_like(kperp_rhoi_s), high_val_s], default=np.nan)

    # 16-84パーセンタイル（約±1σ）をエラーバンドとする
    VB_low, VB_med, VB_high = np.nanpercentile(VB_samples, [16, 50, 84], axis=0)
    ERMHD_low, ERMHD_med, ERMHD_high = np.nanpercentile(ERMHD_samples, [16, 50, 84], axis=0)

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    ax1.errorbar(freq, VA2_Bperp2_slice, yerr=(mean_v_A**2)*B_perp2_err*1e6, fmt='o', ms=3, color='b', label=r'$v_\mathrm{A}^{2}B_\perp^{2}$ PSD')
    ax1.errorbar(freq, E_perp2_slice, yerr=E_perp2_err, fmt='o', ms=3, color='orange', label=r'$E_\perp^{2}$ PSD')
    ax1.axvline(x=1/8, c='purple', ls='--', lw=1, label=r'1 $\times$ spin tone')
    ax1.axvline(x=4/8, c='green', ls='--', lw=1, label=r'4 $\times$ spin tone')
    ax1.axvspan(1/8, 4/8, color='gray', alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]')
    ax1.set_title(f'Avg PSD @ {str(t_start)}')
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.set_ylim(1e-6, 1e3)
    ax1.set_xlim(1e-2, 1e2)
    ax1.legend(fontsize=10)

    ax2.errorbar(freq, ratio_dimless, yerr=ratio_err, color='k', fmt='o', ms=3, ls='', label=r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$', zorder=5)
    ax2.set_yscale('log'); ax2.set_xscale('log')
    ax2.plot(freq_th, VB_med, color='green', label=r'dispersion relation (V-M)')
    ax2.fill_between(freq_th, VB_low, VB_high, color='green', alpha=0.2)
    ax2.plot(freq_th, ERMHD_med, color='blue', label=r'dispersion relation (ERMHD)')
    ax2.fill_between(freq_th, ERMHD_low, ERMHD_high, color='blue', alpha=0.2)
    ax2.axvline(x=1/8, c='purple', ls='--', lw=1, label=r'1 $\times$ spin tone')
    ax2.axvline(x=4/8, c='green', ls='--', lw=1, label=r'4 $\times$ spin tone')
    ax2.axvspan(1/8, 4/8, color='gray', alpha=0.3)
    
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$')
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.set_ylim(1e-1, 1e3)
    ax2.set_xlim(1e-2, 1e2)

    plt.tight_layout()
    
    return fig