#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ========= ユーザー設定 =========
BASE_DIR = r'/mnt/j/observation_data/themis/tha/idl_output/per_bin'  # ← ここを環境に合わせて
BIN_EDGES_CSV = os.path.join(BASE_DIR, 'bin_edges.csv')
BIN_FILE_PATTERN = os.path.join(BASE_DIR, 'tha_peef_an_eflux_pa_bin*.csv')

# プロット設定
mpl.rcParams['font.size'] = 14

# デフォルトで作るエネルギー帯 (eV)
BANDS = [
    ('30–300 eV',    30.0,   50.0),
    ('50–100 eV',    50.0,   100.0),
    ('100–200 eV',   100.0,  200.0),
    ('200–300 eV',   200.0,  300.0),
    ('300–500 eV',   300.0,  500.0),
    ('500–1000 eV',  500.0, 1000.0),
    ('1000–2000 eV', 1000.0, 2000.0),
    ('2000–3000 eV', 2000.0, 3000.0),
    ('3000–5000 eV', 3000.0, 5000.0),
    ('5000–10000 eV', 5000.0, 10000.0),
]

# ========= ユーティリティ =========

def load_energy_bins_from_csv(edges_csv_path: str):
    """
    IDLで書き出した bin_edges.csv（Elo,Ehi 列）から
    edges(長さ N+1) と ecent(長さ N) を返す。
    """
    df = pd.read_csv(edges_csv_path)
    # 列名の空白/BOM対策
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

    if not {'Elo', 'Ehi'} <= set(df.columns):
        raise ValueError(f'{edges_csv_path}: 列名に Elo/Ehi が見つからない')

    Elo = df['Elo'].astype(float).to_numpy()
    Ehi = df['Ehi'].astype(float).to_numpy()

    if Elo.shape != Ehi.shape:
        raise ValueError('Elo/Ehi の長さが一致しない')

    # 単調性と正の値をチェック
    if np.any(~np.isfinite(Elo)) or np.any(~np.isfinite(Ehi)):
        raise ValueError('Elo/Ehi に非数が含まれる')
    if np.any(Elo <= 0) or np.any(Ehi <= 0):
        raise ValueError('Elo/Ehi に非正の値が含まれる')
    if np.any(Elo >= Ehi):
        raise ValueError('Elo < Ehi を満たしていない行がある')

    # edges: [Elo0, Ehi0, Ehi1, ..., Ehi_{N-1}]
    edges = np.concatenate([Elo[:1], Ehi])
    # 念のため昇順を保証（必要なら並べ替え）
    if not np.all(np.diff(edges) > 0):
        # 行ごとに昇順に直す（Elo/Ehiが乱れてたらここで整列）
        order = np.argsort(Elo)  # Elo基準
        Elo = Elo[order]
        Ehi = Ehi[order]
        edges = np.concatenate([Elo[:1], Ehi])
        if not np.all(np.diff(edges) > 0):
            raise ValueError('edges が昇順にならない（bin_edges.csv を確認して）')

    # 幾何中心
    ecent = np.sqrt(Elo * Ehi)
    return edges, ecent

def read_bin_edges(bin_edges_csv):
    """bin_edges.csv (Elo,Ehi) を読み込む。返り値: edges (nb+1,), Ecent (nb,)"""
    df = pd.read_csv(bin_edges_csv)
    # 列名ゆらぎ対策
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    assert 'elo' in df.columns and 'ehi' in df.columns, 'bin_edges.csv に Elo,Ehi がない'
    edges = np.empty(len(df) + 1, dtype=float)
    edges[:-1] = df['elo'].values.astype(float)
    edges[1:] = np.maximum(edges[1:], df['ehi'].values.astype(float))  # 念のため昇順強制
    # 幾何平均の中心（参考）
    ecent = np.sqrt(edges[:-1] * edges[1:])
    # 昇順チェック
    if not np.all(np.diff(edges) > 0):
        raise ValueError('Energy edges not strictly increasing.')
    return edges, ecent


def parse_bin_index(path):
    """…_binNNN.csv の NNN を int で返す"""
    m = re.search(r'_bin(\d{3})\.csv$', os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def load_one_bin_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]

    if 'time_iso' not in df.columns:
        raise ValueError(f'{csv_path}: time_iso 列が無い')

    t = pd.to_datetime(df['time_iso'].astype(str).str.strip(),
                       format='%Y-%m-%d/%H:%M:%S.%f',
                       errors='coerce', utc=True)
    na = t.isna()
    if na.any():
        t.loc[na] = pd.to_datetime(df.loc[na, 'time_iso'].astype(str).str.strip(),
                                   format='%Y-%m-%d/%H:%M:%S',
                                   errors='coerce', utc=True)
    t = t.dt.tz_convert(None)
    t = pd.DatetimeIndex(t).unique().sort_values()  # ← Index 化

    PA = np.array([float(c) for c in df.columns[1:]], dtype=float)
    Z  = df.iloc[:, 1:].to_numpy(dtype=float)
    return t, PA, Z





def align_time_and_stack(files, edges_csv_path):
    # エネルギー境界・中心を先に読む
    edges, ecent = load_energy_bins_from_csv(edges_csv_path)

    # 1本目
    t0, PA, Z0 = load_one_bin_csv(files[0])
    t_common = t0
    per_bin = [(t0, Z0)]

    # 残り：共通時刻の積集合を作る
    for f in files[1:]:
        ti, _, Zi = load_one_bin_csv(f)
        t_common = t_common.intersection(ti)
        per_bin.append((ti, Zi))

    # 共通時刻で揃える
    stacked = []
    for ti, Zi in per_bin:
        df = pd.DataFrame(Zi, index=ti)   # (time × pa)
        df = df.reindex(t_common)         # 積集合に合わせる
        stacked.append(df.to_numpy())

    eflux = np.stack(stacked, axis=2)     # (nt, n_pa, n_bin)
    return t_common, PA, edges, ecent, eflux




def eflux_to_nflux(eflux, ecent):
    """
    differential energy flux [eV/(cm^2 s sr eV)] → differential number flux [1/(cm^2 s sr eV)]
    単純に E で割る。eflux: (nt, npa, nbin), ecent: (nbin,)
    """
    E = ecent.reshape((1, 1, -1))
    nflux = eflux / E
    return nflux


def band_reduce(arr_E, edges, bands, mode='mean'):
    """
    エネルギー方向に帯域処理。
    arr_E: (nt, npa, nbin)  （eflux や nflux）
    edges: (nbin+1,)
    bands: list of (label, Elo, Ehi)
    mode: 'mean'（差分: per-eV の平均）or 'integral'（∫dE: 単位から /eV を外す）
    戻り値: dict[label] = (nt, npa) ndarray
    """
    out = {}
    Elo = edges[:-1]
    Ehi = edges[1:]
    dE = Ehi - Elo

    for label, lo, hi in bands:
        # ビン中心が帯域に入るもの、ではなく「ビンが帯域と交差するもの」を厳密に拾う
        # ここでは簡便に、[lo, hi] と [Elo, Ehi] の重なりがあるものに限定
        overlap = ~((Ehi <= lo) | (Elo >= hi))
        idx = np.where(overlap)[0]
        if idx.size == 0:
            out[label] = np.full(arr_E.shape[:2], np.nan)
            continue

        # 重なり長さで重みを付ける（部分ビンも考慮）
        w = np.zeros_like(dE)
        # 重複幅 = min(Ehi, hi) - max(Elo, lo) の正の部分
        w[idx] = np.maximum(0.0, np.minimum(Ehi[idx], hi) - np.maximum(Elo[idx], lo))

        # arr_E の該当ビンだけ抜く
        A = arr_E[:, :, idx]          # (nt, npa, nb_sel)
        W = w[idx].reshape((1, 1, -1))  # (1,1,nb_sel)

        if mode == 'mean':
            # 帯域平均（per-eV のまま保つ）: sum(A * w) / sum(w)
            num = np.nansum(A * W, axis=2)
            den = np.nansum(W, axis=2)
            out_arr = num / den
        elif mode == 'integral':
            # 帯域積分: ∑ A * dE_eff  (重なり幅を dE_eff とする)
            out_arr = np.nansum(A * W, axis=2)
        else:
            raise ValueError("mode must be 'mean' or 'integral'")

        out[label] = out_arr

    return out


def pcolormesh_pad(ax, time_index, pitch_deg, Z, title='', zlog=True, zlabel=''):
    """
    time × pitch の PAD を pcolormesh で描画（Z: (nt, npa)）
    """
    # time を matplotlib の日時数値へ
    t_nums = mpl.dates.date2num(time_index.to_pydatetime())
    # bin edges を作る（time は前後等間隔と仮定して端点を外挿）
    if len(t_nums) > 1:
        dt = np.diff(t_nums).mean()
    else:
        dt = 1.0 / (24 * 60)  # 1分仮定
    t_edges = np.concatenate([t_nums[:1] - dt/2, (t_nums[:-1] + t_nums[1:]) / 2, t_nums[-1:] + dt/2])

    # pitch edges（中心から等間隔と仮定）
    pa = np.asarray(pitch_deg)
    if len(pa) > 1:
        dpa = np.diff(pa).mean()
    else:
        dpa = 180.0
    pa_edges = np.concatenate([pa[:1] - dpa/2, (pa[:-1] + pa[1:]) / 2, pa[-1:] + dpa/2])

    T, P = np.meshgrid(t_edges, pa_edges, indexing='ij')

    Zplot = Z.copy()
    # 値が全て<=0 の場合のログ対策
    if zlog:
        Zplot = np.where(Zplot > 0, Zplot, np.nan)

    pc = ax.pcolormesh(T, P, Zplot, shading='auto')
    if zlog:
        from matplotlib.colors import LogNorm
        pc.set_norm(LogNorm())

    vmin, vmax = 1E2, 1E5
    pc.set_clim(vmin, vmax)
    bar_color = 'turbo'
    pc.set_cmap(plt.get_cmap(bar_color))
    cb = plt.colorbar(pc, ax=ax, pad=0.01)
    if zlabel:
        cb.set_label(zlabel)

    ax.minorticks_on()
    ax.set_ylim(0, 180)
    ax.set_yticks(np.arange(0, 181, 45))
    ax.set_ylabel('Pitch angle [deg]')
    ax.set_title(title)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M:%S'))


def main():
    # ====== 入力パス ======
    base = r'/mnt/j/observation_data/themis/tha/idl_output/per_bin'
    edges_csv = os.path.join(base, 'bin_edges.csv')

    # エネルギー境界・中心をまず読む（本数に合わせてファイル列挙）
    edges, ecent = load_energy_bins_from_csv(edges_csv)
    nbin = len(ecent)

    bin_files = [os.path.join(base, f'tha_peef_an_eflux_pa_bin{i:03d}.csv')
                 for i in range(nbin)]
    bin_files = [f for f in bin_files if os.path.exists(f)]
    if len(bin_files) == 0:
        raise FileNotFoundError(f'per_bin/*.csv が見つからない: {base}')
    if len(bin_files) != nbin:
        print(f'WARN: edges({nbin}) に対して CSV が {len(bin_files)} 個しか無い')

    # ====== CSV 群 → 時刻共通化して 3D スタック (nt × npa × nbin) ======
    # ※ align_time_and_stack は (files, edges_csv_path) で呼ぶ想定
    time_idx, PA, edges, ecent, eflux = align_time_and_stack(bin_files, edges_csv)

    # ====== eflux → number flux (/eV) ======
    nflux = eflux_to_nflux(eflux, ecent)

    # ====== バンド定義（eV） ======
    BANDS = [
    ('30–300 eV',    30.0,   300.0),
    ('300–1000 eV',   300.0,  1000.0),
    ('1000–2000 eV', 1000.0, 5000.0),
    ('30‒5000 eV', 30.0, 5000.0),
]

    # 差分（/eV の平均）と積分（∫dE）の両方を作る
    nflux_mean = band_reduce(nflux, edges, BANDS, mode='mean')      # [/ (cm^2 s sr eV)]
    nflux_int  = band_reduce(nflux, edges, BANDS, mode='integral')  # [/ (cm^2 s sr)]

    # ====== 描画（差分 number flux） ======
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(10, 15), sharex=True)
    if len(BANDS) == 1:
        axes = [axes]

    for ax, (label, _, _) in zip(axes, BANDS):
        Z = nflux_mean[label]
        pcolormesh_pad(ax, time_idx, PA, Z,
                       title=f'Differential number flux (mean) — {label}',
                       zlog=True,
                       zlabel=r'# / (cm$^2$ s sr eV)')
    axes[-1].set_xlabel('UT')
    fig.suptitle('THEMIS/ESA PAD (number flux)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(base, 'pad_number_flux_3bands.png')
    fig.savefig(out_png, dpi=200)
    print(f'Saved: {out_png}')

    # ====== （必要なら）積分量の方も描画 ======
    # fig2, axes2 = plt.subplots(len(BANDS), 1, figsize=(10, 8), sharex=True)
    # if len(BANDS) == 1:
    #     axes2 = [axes2]
    # for ax, (label, _, _) in zip(axes2, BANDS):
    #     Z = nflux_int[label]
    #     pcolormesh_pad(ax, time_idx, PA, Z,
    #                    title=f'Integrated number flux (∫dE) — {label}',
    #                    zlog=True,
    #                    zlabel=r'# / (cm$^2$ s sr)')
    # axes2[-1].set_xlabel('UT')
    # fig2.suptitle('THEMIS/ESA PAD (integrated number flux)')
    # fig2.tight_layout(rect=[0, 0, 1, 0.96])
    # out_png2 = os.path.join(base, 'pad_number_flux_integrated_3bands.png')
    # fig2.savefig(out_png2, dpi=200)
    # print(f'Saved: {out_png2}')


if __name__ == '__main__':
    main()
