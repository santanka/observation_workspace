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

        time_axis = data_dict['Bx_psd'].time.values
        data_dict['time'] = time_axis

        # --- v_A, beta_i をソース別に読み込み（hfa/mid/lepe）---
        for src in ['hfa', 'mid', 'lepe']:
            data_dict[f'v_A_{src}']   = pt.data_quants[f'v_A_{src}'].interp(time=time_axis, method='linear') * 1e3
            data_dict[f'beta_i_{src}'] = pt.data_quants[f'beta_i_{src}'].interp(time=time_axis, method='linear')

        # 互換キー（既存コードのまま動くよう mid をデフォルトに割当て）
        data_dict['v_A_']   = data_dict['v_A_mid']
        data_dict['beta_i'] = data_dict['beta_i_mid']

        data_dict['ion_cyclo_freq'] = pt.data_quants['ion_cyclo_freq'].interp(time=time_axis, method='linear')
        data_dict['V_ion_fac_perp'] = pt.data_quants['v_sys_fac_perp_rolling'].interp(time=time_axis, method='linear') * 1e3
        data_dict['V_th_ion']       = pt.data_quants['V_th_ion'].interp(time=time_axis, method='linear') * 1e3
        data_dict['C_s_ion']        = pt.data_quants['C_s_ion'].interp(time=time_axis, method='linear') * 1e3
        data_dict['tau']            = pt.data_quants['tau'].interp(time=time_axis, method='linear')
        data_dict['r_param']        = data_dict['C_s_ion'] / data_dict['V_th_ion']

        
        # --- 周波数軸の準備とカットオフ ---
        freq = data_dict['Bx_psd'].v.values
        if cutoff_freq is not None:
            if len(cutoff_freq) == 2:
                mask = (freq <= cutoff_freq[0]) | (freq >= cutoff_freq[1])
            elif len(cutoff_freq) == 3:
                mask = ((freq >= cutoff_freq[0]) & (freq <= cutoff_freq[1])) | (freq >= cutoff_freq[2])
            elif len(cutoff_freq) == 4:
                mask = ((freq >= cutoff_freq[0]) & (freq <= cutoff_freq[1])) | ((freq >= cutoff_freq[2]) & (freq <= cutoff_freq[3]))
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
        vth_i = data_dict['V_th_ion'].values
        vperp = data_dict['V_ion_fac_perp'].values
        data_dict['kperp_rhoi'] = (f[None, :] / w_ci[:, None]) * (vth_i[:, None] / vperp[:, None])

    except Exception as e:
        print(f"データの読み込みに失敗しました: {e}")
        return None
        
    return data_dict

def _pick_variant(data_dict, variant='mid'):
    """
    variant in {'mid','hfa','lepe'}
    """
    if variant not in ('mid', 'hfa', 'lepe'):
        raise ValueError("variant must be 'mid', 'hfa', or 'lepe'")
    vA   = data_dict[f'v_A_{variant}']
    beta = data_dict[f'beta_i_{variant}']
    return vA, beta

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
    if use.sum() < 5: return None
    Xl, Yl = np.log10(x[use]), np.log10(y[use])
    try:
        slope, intercept, r_value, p_value, stderr = linregress(Xl, Yl)
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

def _fill_band(ax, x, y_low, y_high, valid, label, color="#5aa8ff", alpha=1):
    """
    x: 1D 配列（freq or k）
    y_low, y_high: 下限・上限
    valid: True の点だけ塗る（False は NaN 扱いで未表示）
    """
    import numpy as np
    y1 = np.where(valid, y_low,  np.nan)
    y2 = np.where(valid, y_high, np.nan)
    ax.fill_between(
        x, y1, y2,
        where=np.isfinite(y1) & np.isfinite(y2),
        facecolor=color, edgecolor='none',
        alpha=alpha,
        label=(label if label is not None else None)
    )

# -----------------------------------------------------------
# 新しいプロット関数: kモード
# -----------------------------------------------------------
def plot_k_spectrum(t_start, data_dict, interval_sec=1, n_samples_mc=500,
                    k_range=(1e-1, 1e2), n_bins=15, fit_range=(3.0, 30.0),
                    variant='mid', include_hfa=True, include_lepe=True):
    """
    1秒間のデータを k 空間でビン詰め。
    variant='mid'|'hfa'|'lepe'
      - mid: 主系列はエラーバーあり
      - hfa/lepe: 主系列でもエラーバーなし
    include_hfa/lepe: True なら比較用に重ね描き（エラーバーなし）
    """
    t_end = t_start + np.timedelta64(interval_sec, 's')
    time_axis = data_dict['time']

    idx_t = (time_axis >= t_start) & (time_axis < t_end)
    if idx_t.sum() < 2:
        print(f"{t_start} のデータが2点未満です。スキップします。")
        return None

    # 1秒分抽出
    Bx_sel = data_dict['Bx_psd'].values[idx_t, :]
    By_sel = data_dict['By_psd'].values[idx_t, :]
    Ex_sel = data_dict['Ex_psd'].values[idx_t, :]
    Ey_sel = data_dict['Ey_psd'].values[idx_t, :]
    k_sel  = data_dict['kperp_rhoi'][idx_t, :]

    # 主系列の vA/beta
    vA_main, beta_main = _pick_variant(data_dict, variant=variant)
    vA2_sel = vA_main.values[idx_t]**2

    # 派生量
    VA2_Bperp2_sel = vA2_sel[:, None] * (Bx_sel + By_sel) * 1e-12
    E_perp2_sel    = Ex_sel + Ey_sel
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_sel = np.sqrt(E_perp2_sel / VA2_Bperp2_sel)

    # ビン詰め
    kmin, kmax = k_range
    edges = np.logspace(np.log10(kmin), np.log10(kmax), n_bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    xerr = np.vstack([centers-edges[:-1], edges[1:]-centers])

    vals_VA2 = bin_fill(k_sel, VA2_Bperp2_sel, edges)
    vals_E   = bin_fill(k_sel, E_perp2_sel,    edges)
    vals_R   = bin_fill(k_sel, ratio_sel,      edges)

    gm_VA2, yerr_VA2 = stats_bins(vals_VA2)
    gm_E,   yerr_E   = stats_bins(vals_E)
    gm_R,   yerr_R   = stats_bins(vals_R)

    yerr_VA2, yerr_E, yerr_R, xerr = map(lambda err: np.clip(err,0,np.inf), [yerr_VA2, yerr_E, yerr_R, xerr])

    # フィット（主系列のみ）
    fit_mask = (centers >= fit_range[0]) & (centers <= fit_range[1])
    res_B = fit_powerlaw_loglog(centers, gm_VA2, yerr=yerr_VA2, mask=fit_mask)
    res_E = fit_powerlaw_loglog(centers, gm_E,   yerr=yerr_E,   mask=fit_mask)

    # 理論（主系列の beta 使用）
    tau_sel  = data_dict['tau'][idx_t]
    beta_sel = beta_main[idx_t]
    r_sel    = data_dict['r_param'][idx_t]

    gm_tau, lo_tau, hi_tau = geom_stats(tau_sel)
    gm_beta, lo_beta, hi_beta = geom_stats(beta_sel)
    gm_r,   lo_r,   hi_r   = geom_stats(r_sel)

    rng = np.random.default_rng()
    tau_s  = draw_lognormal(gm_tau,  lo_tau,  hi_tau,  n_samples_mc, rng)
    beta_s = draw_lognormal(gm_beta, lo_beta, hi_beta, n_samples_mc, rng)
    r_s    = draw_lognormal(gm_r,    lo_r,    hi_r,    n_samples_mc, rng)

    k_th = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 1000)
    VB_samp = (1 + k_th[None,:]**2) / np.sqrt(1 + k_th[None,:]**2 * (1 + r_s[:,None]**2))
    k_high = k_th[k_th > np.sqrt(10.)]
    ER_samp = k_high[None,:] * tau_s[:,None] / np.sqrt((beta_s[:,None]*(1+tau_s[:,None]) + 2*tau_s[:,None])*(1+tau_s[:,None]))
    VB_lo, VB_md, VB_hi = ci_band(VB_samp)
    ER_lo, ER_md, ER_hi = ci_band(ER_samp)

    vals_B  = bin_fill(k_sel, Bx_sel + By_sel, edges)  # B⊥^2
    gm_B, _ = stats_bins(vals_B)

    with np.errstate(divide='ignore', invalid='ignore'):
        Q_sel = 1e6 * np.sqrt(E_perp2_sel / (Bx_sel + By_sel))
    vals_Q  = bin_fill(k_sel, Q_sel, edges)
    gm_Q, _ = stats_bins(vals_Q)

    # 1秒内の平均 vA（hfa/lepe）
    m_v_hfa  = _pick_variant(data_dict, 'hfa')[0].values[idx_t].mean()
    m_v_lepe = _pick_variant(data_dict, 'lepe')[0].values[idx_t].mean()

    # PSD 側の帯（vA^2 B⊥^2）：上限 lepe、下限 hfa
    VA2B2_hfa_bins  = (m_v_hfa**2)  * gm_B * 1e-12
    VA2B2_lepe_bins = (m_v_lepe**2) * gm_B * 1e-12
    valid_psd = np.isfinite(VA2B2_hfa_bins) & np.isfinite(VA2B2_lepe_bins) & \
                (VA2B2_hfa_bins > 0) & (VA2B2_lepe_bins > 0)
    lower_psd = np.minimum(VA2B2_hfa_bins,  VA2B2_lepe_bins)
    upper_psd = np.maximum(VA2B2_hfa_bins,  VA2B2_lepe_bins)

    # 比の帯（√(E/B)/vA）：上限 lepe、下限 hfa（仕様固定）
    ratio_hfa_bins  = gm_Q / m_v_hfa
    ratio_lepe_bins = gm_Q / m_v_lepe
    valid_ratio = np.isfinite(ratio_hfa_bins) & np.isfinite(ratio_lepe_bins) & \
                  (ratio_hfa_bins > 0) & (ratio_lepe_bins > 0)
    lower_r = np.minimum(ratio_hfa_bins, ratio_lepe_bins)
    upper_r = np.maximum(ratio_hfa_bins, ratio_lepe_bins)

    # 連続する valid 区間ごとに分割して塗る（橋渡し防止）
    def _fill_segmented(ax, x, ylo, yhi, valid, label=None):
        N = len(x); i = 0; first = True
        while i < N:
            while i < N and not valid[i]:
                i += 1
            j = i
            while j < N and valid[j]:
                j += 1
            if j - i >= 2:
                _fill_band(ax, x[i:j], ylo[i:j], yhi[i:j],
                           np.ones(j - i, dtype=bool),
                           label if first else None,
                           color="#4ea1ff", alpha=0.60)
                first = False
            i = j

    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1]})

    # 上段：PSD（まず帯を敷く）
    ax1.set_xscale('log'); ax1.set_yscale('log')
    _fill_segmented(ax1, centers, lower_psd, upper_psd, valid_psd,
                    label=r'$v_A^2 B_\perp^2$ range [hfa–lepe]')

    # 主系列（variant）
    main_err = (variant == 'mid')
    if main_err:
        ax1.errorbar(centers, gm_VA2, yerr=yerr_VA2, xerr=xerr, color='blue',   fmt='o', ms=4,
                     label=fr'$v_A^2 B_\perp^2$ PSD [{variant}]')
        ax1.errorbar(centers, gm_E,   yerr=yerr_E,   xerr=xerr, color='orange', fmt='o', ms=4,
                     label=r'$E_\perp^2$ PSD')
    else:
        ax1.plot(centers, gm_VA2, 'o-', ms=4, label=fr'$v_A^2 B_\perp^2$ PSD [{variant}]')
        ax1.plot(centers, gm_E,   'o-', ms=4, label=r'$E_\perp^2$ PSD')

    # フィット線（主系列）
    if res_B:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 100)
        ax1.plot(xx, res_B["model"](xx), '--', color='blue')
        ax1.text(0.05, 0.25, fr'$\kappa_B={res_B["kappa"]:.2f} \pm {res_B["stderr"]:.2f}$',
                 transform=ax1.transAxes, color='blue',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    if res_E:
        xx = np.logspace(np.log10(fit_range[0]), np.log10(fit_range[1]), 100)
        ax1.plot(xx, res_E["model"](xx), '--', color='orange')
        ax1.text(0.05, 0.15, fr'$\kappa_E={res_E["kappa"]:.2f} \pm {res_E["stderr"]:.2f}$',
                 transform=ax1.transAxes, color='orange',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax1.set_ylabel(r'PSD [(mV/m)$^2$/Hz]')
    ax1.set_title(f'Binned PSD @ {str(t_start)}')
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    ax1.set_xlim(kmin, kmax); ax1.set_ylim(1e-6, 1e3)
    ax1.legend(loc='upper right', fontsize=9)

    # 下段：比（まず帯を敷く）
    ax2.set_xscale('log'); ax2.set_yscale('log')
    _fill_segmented(ax2, centers, lower_r, upper_r, valid_ratio,
                    label=r'$\sqrt{E_\perp^2/B_\perp^2}/v_A$ range [hfa–lepe]')

    if main_err:
        ax2.errorbar(centers, gm_R, yerr=yerr_R, xerr=xerr, color='k', fmt='o', ms=4,
                     label=fr'$\sqrt{{E_\perp^2/B_\perp^2}}/v_A$ [{variant}]')
    else:
        ax2.plot(centers, gm_R, 'o-', ms=4, label=fr'$\sqrt{{E_\perp^2/B_\perp^2}}/v_A$ [{variant}]')

    # 理論曲線
    ax2.plot(k_th, VB_md, color='green', lw=1.5, label='dispersion (V-M)')
    ax2.fill_between(k_th, VB_lo, VB_hi, color='green', alpha=0.2)
    ax2.plot(np.logspace(-1, np.log10(np.sqrt(0.1)), 100), np.ones(100), color='blue', lw=1.5, label='dispersion (ERMHD)')
    ax2.plot(k_high, ER_md, color='blue', lw=1.5)
    ax2.fill_between(k_high, ER_lo, ER_hi, color='blue', alpha=0.2)

    ax2.set_xlabel(r'$\omega_{\mathrm{sc}} \rho_{\mathrm{i}} / |V_{\mathrm{flow}\perp}|$')
    ax2.set_ylabel(r'$\sqrt{E_\perp^2/B_\perp^2} / v_A$')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.set_xlim(kmin, kmax); ax2.set_ylim(1e-1, 1e2)
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()

    fit_results = {
        'time': t_start,
        'kappa_B': res_B['kappa'] if res_B else np.nan,
        'kappa_B_err': res_B['stderr'] if res_B else np.nan,
        'kappa_E': res_E['kappa'] if res_E else np.nan,
        'kappa_E_err': res_E['stderr'] if res_E else np.nan
    }
    return fig, fit_results



def plot_freq_spectrum(t_start, data_dict, interval_sec=1, n_samples_mc=500,
                       variant='mid', include_hfa=True, include_lepe=True):
    """
    指定された時刻から1秒間の平均スペクトルを作成。
    variant='mid'|'hfa'|'lepe'
      - mid: エラーバーあり（従来通り）
      - hfa/lepe: エラーバーなし
    include_hfa/lepe: True なら比較用に重ね描き（エラーバーなし）
    """
    t_end = t_start + np.timedelta64(interval_sec, 's')
    freq = data_dict['freq']
    freq_th = np.logspace(-2, 2, 1000)

    try:
        Bx_psd_sec = data_dict['Bx_psd'].sel(time=slice(t_start, t_end))
        if Bx_psd_sec.time.size < 2:
            print(f"{t_start} のデータが2点未満です。スキップします。")
            return None
        By_psd_sec = data_dict['By_psd'].sel(time=slice(t_start, t_end))
        Ex_psd_sec = data_dict['Ex_psd'].sel(time=slice(t_start, t_end))
        Ey_psd_sec = data_dict['Ey_psd'].sel(time=slice(t_start, t_end))
    except Exception as e:
        print(f"データスライスの取得に失敗: {e}")
        return None

    # 時間平均/標準偏差（E,Bは共通）
    mean_Bx, mean_By = np.nanmean(Bx_psd_sec.values, 0), np.nanmean(By_psd_sec.values, 0)
    std_Bx,  std_By  = np.nanstd(Bx_psd_sec.values, 0),  np.nanstd(By_psd_sec.values, 0)
    mean_Ex, mean_Ey = np.nanmean(Ex_psd_sec.values, 0), np.nanmean(Ey_psd_sec.values, 0)
    std_Ex,  std_Ey  = np.nanstd(Ex_psd_sec.values, 0),  np.nanstd(Ey_psd_sec.values, 0)

    B_perp2_slice = (mean_Bx + mean_By) * 1e-18
    E_perp2_slice = (mean_Ex + mean_Ey)
    B_perp2_err   = np.sqrt(std_Bx**2 + std_By**2) * 1e-18
    E_perp2_err   = np.sqrt(std_Ex**2 + std_Ey**2)

    # 主系列の vA/beta（variant で切替）
    vA_sec_main, beta_sec_main = _pick_variant(data_dict, variant=variant)
    v_A_sec = vA_sec_main.sel(time=slice(t_start, t_end))
    mean_v_A, std_v_A = v_A_sec.mean().values, v_A_sec.std().values

    # 理論のための他パラメータ平均
    ion_cyclo_sec = data_dict['ion_cyclo_freq'].sel(time=slice(t_start, t_end))
    mean_ion_cyclo, std_ion_cyclo = ion_cyclo_sec.mean().values, ion_cyclo_sec.std().values
    V_ion_sec = data_dict['V_ion_fac_perp'].sel(time=slice(t_start, t_end))
    mean_V_ion, std_V_ion = V_ion_sec.mean().values, V_ion_sec.std().values
    V_th_sec = data_dict['V_th_ion'].sel(time=slice(t_start, t_end))
    mean_V_th, std_V_th = V_th_sec.mean().values, V_th_sec.std().values
    C_s_sec = data_dict['C_s_ion'].sel(time=slice(t_start, t_end))
    mean_C_s, std_C_s = C_s_sec.mean().values, C_s_sec.std().values

    # 主系列の描画量
    VA2_Bperp2_main = (mean_v_A**2) * B_perp2_slice * 1e6
    ratio_main = np.sqrt(E_perp2_slice * 1e-6 / B_perp2_slice) / mean_v_A

    # エラーバーは mid のときだけ
    show_err = (variant == 'mid')
    if show_err:
        ratio_err = np.sqrt(
            0.25*(E_perp2_err/E_perp2_slice)**2 +
            0.25*(B_perp2_err/B_perp2_slice)**2 +
            (std_v_A/mean_v_A)**2
        ) * ratio_main
        # NaN防御
        ratio_err = np.nan_to_num(ratio_err)
    else:
        ratio_err = None

    # 理論（主系列で計算）
    rng = np.random.default_rng()
    v_A_s = rng.normal(mean_v_A,      std_v_A,      n_samples_mc)
    ion_s = rng.normal(mean_ion_cyclo, std_ion_cyclo, n_samples_mc)
    Vio_s = rng.normal(mean_V_ion,    std_V_ion,    n_samples_mc)
    Vth_s = rng.normal(mean_V_th,     std_V_th,     n_samples_mc)
    Cs_s  = rng.normal(mean_C_s,      std_C_s,      n_samples_mc)

    VB_samples = (1 + (freq_th[None,:]/ion_s[:,None]*Vth_s[:,None]/Vio_s[:,None])**2) / \
                 np.sqrt(1 + (freq_th[None,:]/ion_s[:,None])**2 * ((Cs_s[:,None]/Vio_s[:,None])**2 + (Vth_s[:,None]/Vio_s[:,None])**2))
    tau_s   = (Vth_s/Cs_s)**2/2
    beta_s  = (Vth_s/v_A_s)**2
    kρ_s    = (freq_th[None,:]/ion_s[:,None])*(Vth_s[:,None]/Vio_s[:,None])
    highval = kρ_s*tau_s[:,None]/np.sqrt((beta_s[:,None]*(1+tau_s[:,None])+2*tau_s[:,None])*(1+tau_s[:,None]))
    ERMHD_samples = np.select([kρ_s < np.sqrt(0.1), kρ_s > np.sqrt(10.)],
                              [np.ones_like(kρ_s), highval], default=np.nan)
    VB_low, VB_med, VB_high = np.nanpercentile(VB_samples,   [16,50,84], axis=0)
    ER_lo, ER_md, ER_hi     = np.nanpercentile(ERMHD_samples,[16,50,84], axis=0)

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,7), sharex=True, gridspec_kw={'height_ratios':[1,1]})

    # 上段：PSD
    ax1.set_xscale('log'); ax1.set_yscale('log')

    # --- hfa/lepe の帯（vA^2 B⊥^2 側）----------------------------
    #   lepe を上限、hfa を下限。B⊥^2 は共通、vA だけが異なるのでスカラーでスケール
    m_v_hfa  = _pick_variant(data_dict, 'hfa')[0].sel(time=slice(t_start, t_end)).mean().values
    m_v_lepe = _pick_variant(data_dict, 'lepe')[0].sel(time=slice(t_start, t_end)).mean().values

    VA2B2_hfa  = (m_v_hfa**2)  * B_perp2_slice * 1e6
    VA2B2_lepe = (m_v_lepe**2) * B_perp2_slice * 1e6

    # 欠損（スピントーンのノッチ等）は塗らない
    valid_psd = np.isfinite(VA2B2_hfa) & np.isfinite(VA2B2_lepe) & (VA2B2_hfa > 0) & (VA2B2_lepe > 0)

    lower_psd = np.minimum(VA2B2_hfa, VA2B2_lepe)   # 念のため大小は取り直す
    upper_psd = np.maximum(VA2B2_hfa, VA2B2_lepe)
    spin1, spin4 = 1/8, 4/8
    left  = freq <= spin1
    right = freq >= spin4

    # PSD 側：左だけ凡例、右は凡例なし
    _fill_band(ax1, freq[left],  lower_psd[left],  upper_psd[left],  valid_psd[left],
               label=r'$v_A^2 B_\perp^2$ range [hfa–lepe]')
    _fill_band(ax1, freq[right], lower_psd[right], upper_psd[right], valid_psd[right],
               label=None)

    
    if show_err:
        ax1.errorbar(freq, VA2_Bperp2_main, yerr=(mean_v_A**2)*B_perp2_err*1e6, fmt='o', ms=3, color='b',
                     label=fr'$v_A^2 B_\perp^2$ PSD [{variant}]')
        ax1.errorbar(freq, E_perp2_slice,   yerr=E_perp2_err, fmt='o', ms=3, color='orange',
                     label=r'$E_\perp^2$ PSD')
    else:
        ax1.plot(freq, VA2_Bperp2_main, 'o-', ms=3, label=fr'$v_A^2 B_\perp^2$ PSD [{variant}]')
        ax1.plot(freq, E_perp2_slice,   'o-', ms=3, label=r'$E_\perp^2$ PSD')

    ax1.axvline(x=1/8, c='purple', ls='--', lw=1, label=r'1 $\times$ spin tone')
    ax1.axvline(x=4/8, c='green',  ls='--', lw=1, label=r'4 $\times$ spin tone')
    ax1.axvspan(1/8, 4/8, color='gray', alpha=0.3)
    ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]')
    ax1.set_title(f'Avg PSD @ {str(t_start)}')
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.set_ylim(1e-6, 1e3); ax1.set_xlim(1e-2, 1e2)
    ax1.legend(loc='lower left', fontsize=9)

    # 下段：比
    ax2.set_xscale('log'); ax2.set_yscale('log')

    # --- hfa/lepe の帯（比 √(E⊥²/B⊥²)/vA 側）--------------------
    #   E⊥²/B⊥² は共通。vA の逆比例なので lepe を上限、hfa を下限に「固定で」塗る指定どおりにする
    Q = np.sqrt(E_perp2_slice * 1e-6 / B_perp2_slice)  # = √(E/B)
    ratio_hfa  = Q / m_v_hfa
    ratio_lepe = Q / m_v_lepe
    valid_ratio = np.isfinite(ratio_hfa) & np.isfinite(ratio_lepe) & (ratio_hfa > 0) & (ratio_lepe > 0)

    # 指定に合わせて「上限 lepe、下限 hfa」で塗る（物理的には逆転する区間もあるが仕様優先）
    lower_r = np.minimum(ratio_hfa, ratio_lepe)   # 念のため大小は取り直し
    upper_r = np.maximum(ratio_hfa, ratio_lepe)
    _fill_band(ax2, freq[left],  lower_r[left],  upper_r[left],  valid_ratio[left],
           label=r'$\sqrt{E_\perp^2/B_\perp^2}/v_A$ range [hfa–lepe]')
    _fill_band(ax2, freq[right], lower_r[right], upper_r[right], valid_ratio[right],
           label=None)
    
    if show_err:
        ax2.errorbar(freq, ratio_main, yerr=ratio_err, color='k', fmt='o', ms=3, ls='', zorder=5,
                     label=fr'$\sqrt{{E_\perp^2/B_\perp^2}}/v_A$ [{variant}]')
    else:
        ax2.plot(freq, ratio_main, 'o-', ms=3, label=fr'$\sqrt{{E_\perp^2/B_\perp^2}}/v_A$ [{variant}]')

    ax2.plot(freq_th, VB_med, color='green', label='dispersion (V-M)')
    ax2.fill_between(freq_th, VB_low, VB_high, color='green', alpha=0.2)
    ax2.plot(freq_th, ER_md, color='blue', label='dispersion (ERMHD)')
    ax2.fill_between(freq_th, ER_lo, ER_hi, color='blue', alpha=0.2)

    ax2.axvline(x=1/8, c='purple', ls='--', lw=1, label=r'1 $\times$ spin tone')
    ax2.axvline(x=4/8, c='green',  ls='--', lw=1, label=r'4 $\times$ spin tone')
    ax2.axvspan(1/8, 4/8, color='gray', alpha=0.3)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$')
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.set_ylim(1e-1, 1e2); ax2.set_xlim(1e-2, 1e2)
    ax2.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    return fig