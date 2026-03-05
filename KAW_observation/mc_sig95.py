#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

JOBLIB_TMP = "/home/satanka/joblib_tmp"
os.makedirs(JOBLIB_TMP, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = JOBLIB_TMP
os.environ["TMPDIR"] = JOBLIB_TMP


# uniform bins in [0,1]
BINS = np.linspace(0.0, 1.0, 1001, dtype=np.float32)

def set_thread_env(n: int = 1) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

def generate_red_noise(N: int, g: float = 0.72, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(N).astype(np.float32)
    red = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        red[i] = g * red[i - 1] + noise[i]
    return red

def _mc_worker_to_hist(idx, tw, fs, n_points, g, dt, s0, dj, J, block_f):
    d1 = generate_red_noise(n_points, g=g, seed=idx)
    d2 = generate_red_noise(n_points, g=g, seed=idx + 10_000_000)

    t = (np.arange(n_points, dtype=np.float32) * dt).astype(np.float32)

    # coef: (Time, Freq)
    _, freqs, coef_e, _, _ = tw.cwt_1d(t, d1, dt=dt, s0=s0, dj=dj, J=J, method="fft")
    _, _,     coef_b, _, _ = tw.cwt_1d(t, d2, dt=dt, s0=s0, dj=dj, J=J, method="fft")

    h = tw.wco_hist_streaming_from_cwtcoef(
        coef_e, coef_b, dt, dj, freqs, n_points, BINS, block_f=block_f
    )
    return h

def run_wco_monte_carlo(tw, fs, n_iterations, n_points, g, n_jobs, J, block_f, batch_size):
    dt, s0, dj = 1.0 / fs, 2.0, 1.0 / 32.0

    if J is None:
        J = int(np.ceil(np.log2((n_points * dt / 3.0) / s0) / dj))

    # freq axis once
    t_dummy = np.arange(256, dtype=np.float32) * dt
    _, freqs, _, _, _ = tw.cwt_1d(
        t_dummy, np.zeros_like(t_dummy), dt=dt, s0=s0, dj=dj, J=J, method="fft"
    )
    n_freqs = len(freqs)

    total_hists = np.zeros((n_freqs, len(BINS) - 1), dtype=np.int64)

    print(
        f"MC(full freq): fs={fs}Hz, N={n_points}, J={J}, F={n_freqs}, "
        f"iters={n_iterations}, jobs={n_jobs}, block_f={block_f}, batch={batch_size}"
    )

    with tqdm(total=n_iterations, desc="Monte Carlo Simulation") as pbar:
        for i0 in range(0, n_iterations, batch_size):
            current = min(batch_size, n_iterations - i0)

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_mc_worker_to_hist)(
                    i0 + j, tw, fs, n_points, g, dt, s0, dj, J, block_f
                )
                for j in range(current)
            )

            for h in results:
                total_hists += h

            pbar.update(current)

    sig95 = np.full(n_freqs, np.nan, dtype=np.float32)
    for f_idx in range(n_freqs):
        s = total_hists[f_idx].sum()
        if s == 0:
            continue
        cdf = np.cumsum(total_hists[f_idx]) / s
        idx95 = np.searchsorted(cdf, 0.95)
        idx95 = min(idx95, len(BINS) - 2)
        sig95[f_idx] = BINS[idx95]

    return freqs.astype(np.float32), sig95.astype(np.float32)

def main(
    *,
    root: str | Path,
    cache_dir: str | Path,
    fs: float,
    n_points: int,
    J: int | None = None,
    n_iterations: int = 1000,
    g: float = 0.72,
    n_jobs: int = -1,
    block_f: int = 32,
    batch_size: int = 32,
    threads: int = 1,
    tag: str = "",
    overwrite: bool = False,
):
    """
    main(fs=..., ...) 形式で呼べるエントリポイント。
    - root: observation_workspace (module_handmade が見える場所)
    - cache_dir: sig95 を保存するフォルダ
    - tag: ファイル名の識別子（例: seg3）
    """

    set_thread_env(threads)

    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import importlib
    import module_handmade.tdwavelet_themis as tw
    importlib.reload(tw)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jstr = f"J{J}" if J is not None else "Jauto"
    tag_str = f"_{tag}" if tag else ""
    out = cache_dir / f"sig95_fs{fs:.4f}{tag_str}_{jstr}.nc"

    if out.exists() and not overwrite:
        print(f"Cache exists (skip): {out}")
        return out

    freqs, sig95 = run_wco_monte_carlo(
        tw=tw,
        fs=fs,
        n_iterations=n_iterations,
        n_points=n_points,
        g=g,
        n_jobs=n_jobs,
        J=J,
        block_f=block_f,
        batch_size=batch_size,
    )

    ds = xr.Dataset({"sig95": ("freq", sig95)}, coords={"freq": freqs})
    ds.to_netcdf(out)
    print(f"Saved: {out}")
    return out

if __name__ == "__main__":
    out = main(
        root="/home/satanka/Documents/observation_workspace",
        cache_dir="/mnt/j/observation_data/Arase_analysis_save_data/mc_cache",
        fs=64.0,
        n_points=209017,
        J=373,
        n_iterations=1000,
        n_jobs=8,
        block_f=32,
        batch_size=8,
        threads=1,
        tag="seg1",
    )
