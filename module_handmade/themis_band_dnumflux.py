# Create a reusable module that reads THEMIS peef/peif per-bin eflux files from a folder,
# computes band-AVERAGED differential number flux [/cm^2 sr s eV] over a specified energy band,
# and writes a CSV next to the inputs.
#
# Usage example:
#   from themis_band_dnumflux import compute_band_dnumflux
#   compute_band_dnumflux(
#       folder="/mnt/j/observation_data/themis/tha/idl_output",
#       spec="peef",          # or "peif"
#       Emin=14.0, Emax=20.0,
#       save_name="tha_peef_dnumflux_E14to20.csv"  # optional; default auto name
#   )
#
# The script expects:
#   - {folder}/bin_edges_{spec}.csv  (columns: Elo,Ehi; index 0 -> bin001)
#   - {folder}/tha_{spec}_an_eflux_pa_bin{bin:03d}.csv (columns: time_iso, <pitch angle centers in deg>)
# Output CSV:
#   - time_iso + the same pitch-angle columns, values are band-averaged differential number flux [/cm^2 sr s eV]
#
import os
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

__all__ = ["compute_band_dnumflux"]

def _read_edges(folder: str, spec: str) -> pd.DataFrame:
    path = os.path.join(folder, f"bin_edges_{spec}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"bin edges not found: {path}")
    df = pd.read_csv(path)
    if not {"Elo","Ehi"}.issubset(df.columns):
        raise ValueError(f"bin_edges_{spec}.csv must have Elo,Ehi columns; got: {df.columns.tolist()}")
    df = df.reset_index(drop=True)
    df["bin"] = np.arange(len(df)) + 1
    df["width"] = df["Ehi"] - df["Elo"]
    # geometric-mean energy per bin for converting energy flux -> number flux density
    df["Emid"] = np.sqrt(np.maximum(df["Elo"], 1e-300) * np.maximum(df["Ehi"], 1e-300))
    return df

def _overlaps(edges: pd.DataFrame, Emin: float, Emax: float) -> List[Tuple[int, float]]:
    Emin, Emax = float(Emin), float(Emax)
    if not (np.isfinite(Emin) and np.isfinite(Emax) and Emax > Emin):
        raise ValueError("Emin < Emax must hold and be finite.")
    out = []
    for i, row in edges.iterrows():
        lo = max(row["Elo"], Emin)
        hi = min(row["Ehi"], Emax)
        dE = max(0.0, hi - lo)
        if dE > 0:
            out.append((i, dE))
    return out

def _coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def compute_band_dnumflux(
    folder: str,
    spec: str,
    Emin: float,
    Emax: float,
    save_name: str = None
) -> str:
    """
    Compute band-AVERAGED differential number flux [/cm^2 sr s eV] over [Emin, Emax]
    for THEMIS (tha) peef/peif per-bin files in 'folder'.

    Parameters
    ----------
    folder : str
        Directory containing 'bin_edges_{spec}.csv' and 'tha_{spec}_an_eflux_pa_binNNN.csv'.
    spec : {'peef','peif'}
        Dataset key (electron/ion).
    Emin, Emax : float
        Energy range [eV]. Must satisfy Emin < Emax.
    save_name : str, optional
        Output CSV file name to create in 'folder'. If None, an auto name is used.

    Returns
    -------
    out_path : str
        Path to the written CSV.
    """
    spec = spec.lower()
    if spec not in ("peef", "peif"):
        raise ValueError("spec must be 'peef' or 'peif'")

    edges = _read_edges(folder, spec)
    ov = _overlaps(edges, Emin, Emax)
    if not ov:
        raise ValueError("指定のエネルギー範囲に重なるbinがありません。")

    band_width = Emax - Emin
    # weights for band-AVERAGED density
    weights = {i: dE / band_width for i, dE in ov}

    time_col = "time_iso"
    averaged = None
    time_vals = None
    pa_cols: List[str] = None

    # iterate overlapped bins
    for i, dE in ov:
        bin_no = i + 1
        fpath = os.path.join(folder, f"tha_{spec}_an_eflux_pa_bin{bin_no:03d}.csv")
        if not os.path.exists(fpath):
            # allow missing; just skip
            continue
        df = pd.read_csv(fpath)
        if time_col not in df.columns:
            raise ValueError(f"{fpath} に 'time_iso' 列が必要です。")

        # fix column order and types
        if pa_cols is None:
            pa_cols = [c for c in df.columns if c != time_col]
        df = df[[time_col] + pa_cols].copy()
        df = _coerce_numeric(df, exclude=[time_col])

        # convert differential energy flux -> differential number flux density
        Emid = float(edges.loc[i, "Emid"])
        num_density = df[pa_cols] / Emid  # [/cm^2 sr s eV]

        # accumulate weighted average
        w = weights[i]
        contrib = num_density * w
        if averaged is None:
            averaged = contrib.copy()
            time_vals = df[time_col]
        else:
            averaged = averaged.add(contrib, fill_value=0.0)

    if averaged is None:
        raise FileNotFoundError("重なりbinのCSVが見つかりませんでした。")

    out_df = pd.concat([time_vals, averaged], axis=1).sort_values(time_col).reset_index(drop=True)

    # default name
    if save_name is None:
        Emin_s = f"{Emin:g}".replace(".", "p")
        Emax_s = f"{Emax:g}".replace(".", "p")
        save_name = f"tha_{spec}_dnumflux_E{Emin_s}to{Emax_s}.csv"

    out_path = os.path.join(folder, save_name)
    out_df.to_csv(out_path, index=False)
    return out_path

# Optional small CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute band-AVERAGED differential number flux for THEMIS peef/peif.")
    ap.add_argument("--folder", required=True, help="Input/output folder path")
    ap.add_argument("--spec", required=True, choices=["peef","peif"], help="Dataset type")
    ap.add_argument("--emin", required=True, type=float, help="Emin [eV]")
    ap.add_argument("--emax", required=True, type=float, help="Emax [eV]")
    ap.add_argument("--save_name", default=None, help="Output CSV file name")
    args = ap.parse_args()
    p = compute_band_dnumflux(args.folder, args.spec, args.emin, args.emax, args.save_name)
    print(p)
