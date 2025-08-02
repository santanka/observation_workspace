# psd_plotter.py

import numpy as np
import matplotlib.pyplot as plt
import pytplot as pt

def load_and_prepare_data(cutoff_freq=1/6):
    """
    pytplotから必要なデータを読み込み、前処理を行う。
    戻り値: データと周波数軸を含む辞書
    """
    data_dict = {}
    try:
        # 生データを取得
        data_dict['Bx_psd'] = pt.data_quants['B_fac_x_hp_clean']
        data_dict['By_psd'] = pt.data_quants['B_fac_y_hp_clean']
        data_dict['Ex_psd'] = pt.data_quants['E_fac_x_hp_clean']
        data_dict['Ey_psd'] = pt.data_quants['E_fac_y_hp_clean']
        data_dict['v_A_'] = pt.data_quants['Alfven_speed'] * 1E3
        data_dict['ion_cyclo_freq'] = pt.data_quants['ion_cyclo_freq']
        data_dict['V_ion_fac_perp'] = pt.data_quants['erg_lepi_l2_3dflux_FPDU_velocity_fac_perp']
        data_dict['V_th_proton'] = pt.data_quants['V_th_proton']
        data_dict['C_s_proton'] = pt.data_quants['C_s_proton']
        
        # 周波数軸の準備とカットオフ
        freq = data_dict['Bx_psd'].v.values
        mask = freq >= cutoff_freq
        data_dict['freq'] = freq[mask]
        
        # 各PSDデータにマスクを適用
        for key in ['Bx_psd', 'By_psd', 'Ex_psd', 'Ey_psd']:
            data_dict[key] = data_dict[key].isel(v_dim=mask)

    except Exception as e:
        print(f"データの読み込みに失敗しました: {e}")
        return None
        
    return data_dict

def plot_averaged_spectrum(t_start, data_dict, interval_sec=1, n_samples_mc=500):
    """
    指定された時刻から1秒間の平均スペクトルと分散（エラーバー/バンド）を計算し、プロットを作成する。
    """
    t_end = t_start + np.timedelta64(interval_sec, 's')
    freq = data_dict['freq']

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
    
    # 各パラメータの分布からランダムサンプリング
    v_A_s = rng.normal(mean_v_A, std_v_A, size=n_samples_mc)
    ion_cyclo_s = rng.normal(mean_ion_cyclo, std_ion_cyclo, size=n_samples_mc)
    V_ion_s = rng.normal(mean_V_ion, std_V_ion, size=n_samples_mc)
    V_th_s = rng.normal(mean_V_th, std_V_th, size=n_samples_mc)
    C_s_s = rng.normal(mean_C_s, std_C_s, size=n_samples_mc)
    
    # サンプルごとに理論曲線を計算 (結果は (n_samples_mc, n_freq) の2次元配列)
    VB_samples = (1 + (freq[None, :] / ion_cyclo_s[:, None] * V_th_s[:, None] / V_ion_s[:, None])**2) / \
                 np.sqrt(1 + (freq[None, :] / ion_cyclo_s[:, None])**2 * ((C_s_s[:, None] / V_ion_s[:, None])**2 + (V_th_s[:, None] / V_ion_s[:, None])**2))

    tau_s = (V_th_s / C_s_s)**2 / 2
    beta_i_s = (V_th_s / v_A_s)**2
    kperp_rhoi_s = (freq[None, :] / ion_cyclo_s[:, None]) * (V_th_s[:, None] / V_ion_s[:, None])
    high_val_s = kperp_rhoi_s * tau_s[:, None] / np.sqrt((beta_i_s[:, None] * (1 + tau_s[:, None]) + 2 * tau_s[:, None]) * (1 + tau_s[:, None]))
    ERMHD_samples = np.select([kperp_rhoi_s < np.sqrt(0.1), kperp_rhoi_s > np.sqrt(10.0)], [np.ones_like(kperp_rhoi_s), high_val_s], default=np.nan)

    # 16-84パーセンタイル（約±1σ）をエラーバンドとする
    VB_low, VB_med, VB_high = np.nanpercentile(VB_samples, [16, 50, 84], axis=0)
    ERMHD_low, ERMHD_med, ERMHD_high = np.nanpercentile(ERMHD_samples, [16, 50, 84], axis=0)

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    ax1.errorbar(freq, VA2_Bperp2_slice, yerr=(mean_v_A**2)*B_perp2_err*1e6, fmt='o', ms=3, color='b', label=r'$v_\mathrm{A}^{2}B_\perp^{2}$ PSD')
    ax1.errorbar(freq, E_perp2_slice, yerr=E_perp2_err, fmt='o', ms=3, color='orange', label=r'$E_\perp^{2}$ PSD')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'PSD [$(\mathrm{mV/m})^{2}$/Hz]')
    ax1.set_title(f'Avg PSD @ {str(t_start)}')
    ax1.grid(True, which='both', ls='--', lw=0.5)
    ax1.set_ylim(1e-4, 5e3)
    ax1.set_xlim(1/8, 4e1)
    ax1.legend(fontsize=10)

    ax2.errorbar(freq, ratio_dimless, yerr=ratio_err, color='k', fmt='o', ms=3, ls='', label=r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$', zorder=5)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.plot(freq, VB_med, color='green', label=r'dispersion relation (V-M)')
    ax2.fill_between(freq, VB_low, VB_high, color='green', alpha=0.2)
    ax2.plot(freq, ERMHD_med, color='blue', label=r'dispersion relation (ERMHD)')
    ax2.fill_between(freq, ERMHD_low, ERMHD_high, color='blue', alpha=0.2)
    
    ax2.legend(fontsize=10)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'$\sqrt{E_\perp^{2}/B_\perp^{2}}/v_{\mathrm{A}}$')
    ax2.grid(True, which='both', ls='--', lw=0.5)
    ax2.set_ylim(5e-1, 1e2)
    ax2.set_xlim(1/8, 4e1)

    plt.tight_layout()
    
    return fig