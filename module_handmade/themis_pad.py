import numpy as np
import pytplot as pt
from typing import List, Tuple, Optional

# ---------------------------
# ユーティリティ
# ---------------------------
def _time64(a):  # datetime64 -> int(ns)
    return a.astype('datetime64[ns]').astype('int64')

def _interp_linear_to(dst_t, src_t, src_y):
    """各成分独立の線形補間（src_t: (Ns,), src_y: (Ns, C) or (Ns,)）"""
    t_dst = _time64(dst_t); t_src = _time64(src_t)
    if src_y.ndim == 1:
        return np.interp(t_dst, t_src, src_y)
    out = np.empty((len(t_dst), src_y.shape[1]), dtype=float)
    for i in range(src_y.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, src_y[:, i])
    return out

def _pick_coord(dq, candidates):
    for c in candidates:
        if c in dq.coords:
            return np.asarray(dq.coords[c]), c
    return None, None

def _energy_axis(dq):
    # ESAは eV が多い。coords名は環境差があるので手広く
    return _pick_coord(dq, ['energy','en','spec_bins','E','v','v1','e_bins'])

def _theta_phi_axes(dq, theta_var=None, phi_var=None):
    # まず DataArray の coords から探索
    theta, _ = _pick_coord(dq, ['theta','THETA','elev','ELE','v2','angle'])
    phi,   _ = _pick_coord(dq, ['phi','PHI','azim','AZI','v3','gyro'])
    # 外から別変数で渡された場合はそれを優先（tplot 1D配列）
    if theta_var is not None:
        xt, yt = pt.get_data(theta_var); theta = yt
    if phi_var is not None:
        xp, yp = pt.get_data(phi_var);   phi   = yp
    return theta, phi

def _unit_vec_from_theta_phi(theta_deg, phi_deg):
    """
    想定: DSL系、theta=極角(0°は+Z=スピン軸)、phi=スピン面の方位。
    データ定義が異なる場合はここを合わせること。
    """
    th = np.deg2rad(theta_deg); ph = np.deg2rad(phi_deg)
    ux = np.sin(th) * np.cos(ph)
    uy = np.sin(th) * np.sin(ph)
    uz = np.cos(th)
    return ux, uy, uz

def _flatten_angle_grid(theta, phi):
    """
    theta, phi が 1Dなら meshgrid、2Dならそのまま。
    返り値: (ux,uy,uz) 各 (Na,)
    """
    if theta.ndim == 1 and phi.ndim == 1:
        PHI, TH = np.meshgrid(phi, theta, indexing='xy')  # (Ntheta, Nphi)
    else:
        TH, PHI = np.asarray(theta), np.asarray(phi)
    ux, uy, uz = _unit_vec_from_theta_phi(TH, PHI)
    return ux.ravel(), uy.ravel(), uz.ravel()

def _angle_reduce(z_ne_a, alpha_deg, pa_bins, energy_band=None, energy_axis=None,
                  pitch_range=None, combine='sum'):
    """
    z_ne_a: (Nt, Ne, Na) の 3D分布
    alpha_deg: (Nt, Na) の pitch角度
    pa_bins: 1D bin（度）
    energy_band: (emin, emax) eV 範囲 or None
    energy_axis: (Ne,) エネルギー中心 eV
    pitch_range: (pa_min, pa_max) or None
    combine: 'sum' or 'mean'（角度平均→エネ方向の合成）
    返り: PAD: (Nt, Npa)
    """
    Nt, Ne, Na = z_ne_a.shape
    if energy_axis is None:
        e_mask = np.ones(Ne, dtype=bool)
    else:
        if energy_band is None:
            e_mask = np.ones(Ne, dtype=bool)
        else:
            emin, emax = energy_band
            e_mask = (energy_axis >= emin) & (energy_axis <= emax)
            if e_mask.sum() == 0:
                return np.full((Nt, len(pa_bins)-1), np.nan)
    Z = z_ne_a[:, e_mask, :]  # (Nt, Nb, Na)

    if pitch_range is None:
        mask_global = np.ones_like(alpha_deg, dtype=bool)
    else:
        pa_min, pa_max = pitch_range
        mask_global = (alpha_deg >= pa_min) & (alpha_deg <= pa_max)

    Npb = len(pa_bins)-1
    PAD = np.full((Nt, Npb), np.nan)
    # z_ne_a を (Nt, Na, Nb) に入れ替えて Na 次元を先に扱いやすくする
    Z_tan = np.swapaxes(Z, 1, 2)  # (Nt, Na, Nb)

    for ib in range(Npb):
        amin, amax = pa_bins[ib], pa_bins[ib+1]
        mask = mask_global & (alpha_deg >= amin) & (alpha_deg < amax)  # (Nt, Na)
        # 角度平均
        w = mask[..., None]                         # (Nt, Na, 1)
        num = np.where(w, Z_tan, 0.0).sum(axis=1)  # (Nt, Nb)
        den = mask.sum(axis=1)[:, None]            # (Nt, 1)
        ang_avg = np.where(den > 0, num/np.maximum(den, 1), np.nan)  # (Nt, Nb)
        # エネルギー結合
        if combine == 'mean':
            val = np.nanmean(ang_avg, axis=1)      # (Nt,)
        else:
            val = np.nansum(ang_avg, axis=1)       # (Nt,)
        PAD[:, ib] = val
    return PAD

def _prefix_from_dist(dist_var):
    # IDL流の命名に寄せるための prefix 推定（緩め）
    # 例: 'tha_peef_3dflux' -> 'tha_peef'
    if '_3d' in dist_var:
        return dist_var.split('_3d')[0]
    if '_en_' in dist_var:
        return dist_var.split('_en_')[0]
    return dist_var

# ---------------------------
# 本体：SPEDAS風 Python 版
# ---------------------------
def thm_part_products_py(
    dist_var: str,
    outputs: List[str] = ('pa', 'energy'),
    pitch: Optional[Tuple[float, float]] = (0.0, 180.0),
    phi:   Optional[Tuple[float, float]] = None,
    theta: Optional[Tuple[float, float]] = None,
    gyro:  Optional[Tuple[float, float]] = None,     # 未実装（必要なら phi を流用）
    energy: Optional[Tuple[float, float]] = None,     # eV: 全体レンジの切り出し
    energy_bands: Optional[List[Tuple[float,float]]] = None,  # eV: バンドごとに PAD 作成
    units: str = 'eflux',                             # ここでは表示ラベル用途。変換は未実装
    mag_name: Optional[str] = None,                   # 例: 'tha_fgs_dsl'（必須: PAに必要）
    coord: str = 'dsl',                               # 表示ラベル用途
    pa_bins: np.ndarray = np.arange(0, 181, 10),      # 度
    theta_var: Optional[str] = None,                  # 角度座標を外部変数で渡す場合
    phi_var: Optional[str] = None,
    combine: str = 'sum',                             # エネ方向の合成方法
    suffix: str = ''
) -> List[str]:
    """
    THEMIS L1 ESA 3D分布 (dist_var) → SPEDAS風の PAD/ENERGY スペクトルを作成して tplot に保存。
    - dist_var: 角度つき3D（time×energy×angle[×angle]）の tplot 変数（あなたがロード済み）
    - outputs: ['pa','energy', ...] を指定（'phi','theta' も将来拡張可）
    - pitch/phi/theta: 角度範囲 [min,max]（deg）での制限
    - energy: eV で全体の切り出し
    - energy_bands: eV 範囲のリスト。各バンドごとに別PADを作る
    - units/coord: 出力ラベル用
    - mag_name: ピッチ角の計算に必須（Bベクトルの tplot 名）
    返り値: 作成した tplot 変数名のリスト
    """
    created = []
    # 3D分布（DataArray想定）を取得
    dq = pt.data_quants.get(dist_var)
    if dq is None:
        raise ValueError(f"tplot 変数が見つからない: {dist_var}")
    times = dq.time.values
    Z = np.asarray(dq)     # (Nt, Ne, Na) or (Nt, Ne, Nphi, Ntheta)
    dims = list(dq.dims)

    # エネルギー軸
    e_axis, e_name = _energy_axis(dq)
    if e_axis is None:
        raise RuntimeError("エネルギー軸が特定できない（coordsに energy/en/spec_bins 等が無い）")

    # 角度座標
    THETA, PHI = _theta_phi_axes(dq, theta_var, phi_var)
    if THETA is None or PHI is None:
        raise RuntimeError("theta/phi 座標が見つからない。coords か theta_var/phi_var で渡して")

    # 形状を (Nt, Ne, Na) に揃える
    if Z.ndim == 4:
        Z = Z.reshape(Z.shape[0], Z.shape[1], -1)
    elif Z.ndim != 3:
        raise RuntimeError(f"想定外の次元: {Z.shape}（Nt,Ne,Ang か Nt,Ne,phi,theta を期待）")

    Nt, Ne, Na = Z.shape

    # energy 全体切り出し
    if energy is not None:
        emin, emax = energy
        emask = (e_axis >= emin) & (e_axis <= emax)
        if emask.sum() == 0:
            raise RuntimeError("指定 energy 範囲に有効binが無い")
        Z = Z[:, emask, :]
        e_used = e_axis[emask]
    else:
        e_used = e_axis

    # ルック方向の単位ベクトル（Na 本）
    ux, uy, uz = _flatten_angle_grid(THETA, PHI)
    if ux.size != Na:
        # 角度展開とデータの並びが不一致
        raise RuntimeError(f"角度ビン数が不一致: look={ux.size}, data={Na}")

    # B を補間して unit ベクトルへ
    if mag_name is None and ('pa' in outputs):
        raise ValueError("PA を出力するには mag_name（例: 'tha_fgs_dsl'）が必須")
    btx, bB = pt.get_data(mag_name)
    if btx is None or bB is None or bB.ndim < 2 or bB.shape[1] < 3:
        raise RuntimeError(f"磁場ベクトルが不正: {mag_name}")
    B = _interp_linear_to(times, btx, bB[:, :3])
    Bn = np.linalg.norm(B, axis=1, keepdims=True); Bn[Bn == 0] = np.nan
    b_hat = B / Bn                       # (Nt,3)
    U = np.stack([ux, uy, uz], axis=-1)  # (Na,3)
    cos_pa = np.clip(b_hat @ U.T, -1.0, 1.0)   # (Nt,Na)
    alpha = np.degrees(np.arccos(cos_pa))      # (Nt,Na)

    prefix = _prefix_from_dist(dist_var)
    # ---------------------------
    # ENERGY スペクトル（角度総和）
    # ---------------------------
    if any(o.lower() == 'energy' for o in outputs):
        # (Nt, Ne, Na) → 角度平均 or 総和
        if combine == 'mean':
            en_spec = np.nanmean(Z, axis=2)       # (Nt, Ne)
        else:
            en_spec = np.nansum(Z, axis=2)        # (Nt, Ne)
        en_name = f"{prefix}_en_{units}{suffix}"
        pt.store_data(en_name, data={'x': times, 'y': e_used, 'z': en_spec})
        pt.options(en_name, 'spec', 1); pt.options(en_name, 'zlog', 1)
        pt.options(en_name, 'ylog', 1); pt.options(en_name, 'ytitle', 'Energy [eV]')
        created.append(en_name)

    # ---------------------------
    # PA スペクトル
    # ---------------------------
    if any(o.lower() == 'pa' for o in outputs):
        bands = energy_bands if energy_bands else [(float(e_used.min()), float(e_used.max()))]
        pa_cent = 0.5 * (pa_bins[:-1] + pa_bins[1:])
        for (bmin, bmax) in bands:
            PAD = _angle_reduce(
                Z, alpha, pa_bins,
                energy_band=(bmin, bmax),
                energy_axis=e_used,
                pitch_range=pitch,
                combine=combine
            )
            lab_e = f"{int(bmin)}-{int(bmax)}eV" if bmax < 1e4 else f"{int(bmin)}-{bmax/1000:.1f}keV"
            pa_name = f"{prefix}_an_{units}_pa_{lab_e}{suffix}"
            pt.store_data(pa_name, data={'x': times, 'y': pa_cent, 'z': PAD})
            pt.options(pa_name, 'spec', 1); pt.options(pa_name, 'zlog', 1)
            pt.options(pa_name, 'ytitle', 'Pitch angle [deg]'); pt.options(pa_name, 'ylog', 0)
            created.append(pa_name)

    # 将来: phi/theta/gyro の角度スペクトルはここに追加（alpha→phi/theta のbinningに置換）

    return created
