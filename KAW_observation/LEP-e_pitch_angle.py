import os

os.environ["ERG_DATA_DIR"] = "/mnt/j/observation_data/"

import matplotlib as mpl
mpl.use('Agg')                               # GUI 無し高速描画

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pyspedas as psp, pytplot as pt
from matplotlib.ticker import MultipleLocator
from concurrent.futures import ProcessPoolExecutor   # I/O バウンド

# -------------------- 0. 基本設定 ----------------------------------
mpl.rcdefaults()
mpl.rcParams['text.usetex']        = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.serif']         = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset']   = 'cm'
plt.rcParams['font.size']          = 28

OUT_DIR = '/mnt/j/KAW_observation/LEP-e_pitch_angle_each_time/20220901/22-24/'
os.makedirs(OUT_DIR, exist_ok=True)


mpl.rcdefaults(); mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 28
})

# ---------- 1. データ読込み ----------
tr = ['20220901/22:25:00', '20220901/23:35:00']
pt.del_data('*')
psp.erg.lepe(tr, datatype='3dflux', level='l2')
psp.erg.mgf (tr, datatype='64hz',  level='l2', coord='dsi')
psp.erg.orb (tr, datatype='def',   level='l2')

lep_var  = 'erg_lepe_l2_3dflux_FEDU'
flux_da  = pt.data_quants[lep_var]
energy_eV = flux_da.v1[0, :].values      # 18 bin（eV 単位）

T0 = np.datetime64('2022-09-01T22:30:00')
T1 = np.datetime64('2022-09-01T23:30:00')
times = [t for t in flux_da.time.values if T0 <= t <= T1]

# ---------- 2. 1 スナップショット処理関数 ----------
def one_snapshot(target):

    dt = np.timedelta64(30, 's')
    str_t = lambda t: str(t.astype('datetime64[s]')).replace('-', '').replace('T', '/')
    trange = [str_t(target - dt), str_t(target + dt)]

    # ------- 時系列を xarray.sel で限定 ------------------
    flux_slice = flux_da.sel(time=slice(*trange))     # ← .sel は xarray Dataset/DataArray に対して行う
    if flux_slice.time.size == 0:
        return
    # target に最も近いインデックス
    tid = int(np.abs(flux_slice.time.values - target).argmin())

    data_pairs = []

    for E in energy_eV:
        if not np.isfinite(E) or E > 5e3:
            continue

        psp.erg.erg_lep_part_products(
            lep_var, outputs='pa', trange=trange,
            energy=[np.floor(E), np.ceil(E)],
            mag_name='erg_mgf_l2_mag_64hz_dsi',
            pos_name='erg_orb_l2_pos_dsi', suffix=f'_PA_{target}')

        pa_da = pt.data_quants[f'{lep_var}_pa_PA_{target}']    # dims=(1, pa)
        pitch = pa_da.spec_bins.values[tid, :]             # 1-D
        flux  = pa_da.values[tid, :]# * E
        data_pairs.append((E, pitch, flux))

    if not data_pairs:
        return

    # ---------- 描画 ----------
    fig, ax = plt.subplots(figsize=(10,10))
    cmap  = plt.cm.turbo
    normE = mpl.colors.LogNorm(vmin=100, vmax=5e3)

    for E, pitch, flux in data_pairs:
        ax.plot(pitch, flux, color=cmap(normE(E)))

    ax.set(xlim=(0,180),
           xlabel='Pitch Angle [deg]',
           yscale='log',
           ylabel=r'$\mathrm{e}^-\,\mathrm{flux}$ [\# / s cm$^{2}$ sr eV]',
           title=pd.to_datetime(target).strftime('%Y-%m-%d %H:%M:%S.%f'))
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.grid(True, which='both', alpha=.3)
    fig.colorbar(mpl.cm.ScalarMappable(norm=normE, cmap=cmap),
                 ax=ax, pad=.02).set_label('Energy [eV]')
    fig.tight_layout()

    fname = f"{pd.to_datetime(target).strftime('%Y%m%dT%H%M%S%f')}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=140)
    plt.close(fig)
    pt.del_data(f'{lep_var}_pa_PA_{target}')

#one_snapshot(times[100])

# ---------- 4. I/O バウンドなので ThreadPoolExecutor ----------
parallel_number = os.cpu_count()
with ProcessPoolExecutor(max_workers=parallel_number) as pool:
    list(pool.map(one_snapshot, times, chunksize=8))
print('✓ finished:', len(times), 'snapshots →', OUT_DIR)