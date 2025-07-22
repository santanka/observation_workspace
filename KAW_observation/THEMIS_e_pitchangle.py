import os, numpy as np, pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
from pathlib import Path

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, BoundaryNorm

# -------------------- 0. 基本設定 ----------------------------------
mpl.rcdefaults()
mpl.rcParams['text.usetex']        = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.serif']         = ['Computer Modern Roman']
mpl.rcParams['mathtext.fontset']   = 'cm'
plt.rcParams['font.size']          = 28

OUT_DIR = '/mnt/j/KAW_observation/THEMIS_A_pitch_angle_each_time/20220901/22-24/'
os.makedirs(OUT_DIR, exist_ok=True)

mpl.rcdefaults(); mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 28
})

dir_name = f'/mnt/j/observation_data/THEMIS_A_from_Artemyev'
file_name_ePA_1 = f'{dir_name}/2022-09-01_koseki_a_peef_PAD_22-23.dat'
file_name_ePA_2 = f'{dir_name}/2022-09-01_koseki_a_peef_PAD_23-24.dat'

def load_electron_flux(fname: str,
                       bands=('[30,300]', '[300,1000]', '[1000,5000]'),
                       n_lines_per_block: int = 5) -> xr.DataArray:
    """
    THEMIS ESA ASCII ファイル → electron_flux DataArray へ変換
    ----------------------------------
    dims  : time × band × pitch
    coords: time (datetime64[ns])
            band (str)
            pitch (deg)
    """
    lines = [ln.rstrip() for ln in Path(fname).read_text().splitlines() if ln.strip()]
    if len(lines) % n_lines_per_block != 0:
        raise ValueError('行数がブロック長の倍数になっていません')

    times, flux_blocks = [], []
    for i in range(0, len(lines), n_lines_per_block):
        # 時刻
        times.append(np.datetime64(lines[i].replace('/', 'T'), 'ns'))
        # pitch 角（全ブロック同一前提）
        if i == 0:
            pitch = np.fromstring(lines[i + 1], sep=' ')
        # フラックス
        block = [np.fromstring(lines[i + 2 + k], sep=' ') for k in range(len(bands))]
        flux_blocks.append(np.stack(block))          # (n_band, n_pitch)

    flux = np.stack(flux_blocks)                     # (Nt, n_band, n_pitch)

    da = xr.DataArray(
        flux,
        dims=('time', 'band', 'pitch'),
        coords=dict(
            time  = ('time',  np.asarray(times)),
            band  = ('band',  np.asarray(bands, dtype='U')),
            pitch = ('pitch', pitch)
        ),
        name='electron_flux'
    )
    return da

ds_ePA_1 = load_electron_flux(file_name_ePA_1)
ds_ePA_2 = load_electron_flux(file_name_ePA_2)

ds_ePA_1 = ds_ePA_1.drop_duplicates(dim='time')
ds_ePA_2 = ds_ePA_2.drop_duplicates(dim='time')
ds_ePA   = xr.concat([ds_ePA_1, ds_ePA_2], dim='time').sortby('time')

times = ds_ePA.time.values

def one_snapshot(target):

    # target は datetime64[ns] 型
    # targetにおけるds_ePAのデータを抽出
    ds = ds_ePA.sel(time=target)
    if ds.time.size == 0:
        return

    # データをプロット
    fig = plt.figure(figsize=(12, 10))

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])
    ax = fig.add_subplot(gs[0, 0])

    # 各バンドのフラックスをプロット
    # [30, 300] eV, [300, 1000] eV, [1000, 5000] eVをそれぞれblue, green, redでプロット
    # cmapを作成
    colors = ['blue', 'green', 'red']

    band_bounds = [30, 300, 1000, 5000]                # 4 つ → 3 区間
    cmap_band   = ListedColormap(['blue', 'green', 'red'])
    norm_band   = BoundaryNorm(np.log10(band_bounds), cmap_band.N, clip=True)

    cax = fig.add_subplot(gs[0, 1])
    cb  = mpl.colorbar.ColorbarBase(cax, cmap=cmap_band, norm=norm_band,
                                boundaries=np.log10(band_bounds), ticks=np.log10(band_bounds),
                                spacing='proportional')
    cb.ax.set_yticklabels(band_bounds)
    cb.set_label('[eV]')

    for i, band in enumerate(ds.band.values):
        flux = ds.sel(band=band).values.astype(float)
        flux[flux <= 0] = np.nan
        if np.all(np.isnan(flux)):
            continue
        ax.plot(ds.pitch.values, flux, color=colors[i], label=band)
    ax.set(xlim=(0, 180),
           xlabel='Pitch Angle [deg]',
           yscale='log',
           ylabel=r'$\mathrm{e}^-\,\mathrm{flux}$ [\# / s cm$^{2}$ sr eV]',
           title=pd.to_datetime(target).strftime('%Y-%m-%d %H:%M:%S'))
    ax.xaxis.set_major_locator(MultipleLocator(22.5))
    ax.grid(True, which='both', alpha=.3)
    #ax.legend(title='Energy Band', loc='upper right')
    fig.tight_layout()

    fname = f"{pd.to_datetime(target).strftime('%Y%m%dT%H%M%S')}.png"
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=140)
    plt.close(fig)

#one_snapshot(times[100])

parallel_number = os.cpu_count()
with ProcessPoolExecutor(max_workers=parallel_number) as pool:
    list(pool.map(one_snapshot, times, chunksize=parallel_number))
print('✓ finished:', len(times), 'snapshots →', OUT_DIR)