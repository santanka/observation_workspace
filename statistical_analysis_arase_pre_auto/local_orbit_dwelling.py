"""Read daily ORB CDFs directly; network access is explicit and limited to missing days."""

from pathlib import Path
import re

import cdflib
import numpy as np
import pandas as pd


REMOTE_ROOT = "https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/"


def orbit_inventory(months, spedas_dir):
    result, missing = {}, []
    for month in months:
        start = pd.Timestamp(month + "-01")
        for day in pd.date_range(start, start + pd.offsets.MonthBegin(1), inclusive="left"):
            candidates = []
            for base in [spedas_dir / "ergsc", spedas_dir]:
                candidates.extend((base / f"satellite/erg/orb/def/{day:%Y}").glob(f"erg_orb_l2_{day:%Y%m%d}_v??.cdf"))
            if candidates:
                result[day] = max(candidates, key=lambda p: (p.name, "ergsc" in p.parts))
            else:
                missing.append(day)
    return result, missing


def download_missing_days(missing, spedas_dir):
    """One directory request per year, then one GET per missing day; sequential."""
    import requests
    result = {}
    with requests.Session() as session:
        for year in sorted({day.year for day in missing}):
            relative = f"satellite/erg/orb/def/{year}/"
            listing = session.get(REMOTE_ROOT + relative, timeout=(15, 90))
            listing.raise_for_status()
            names = set(re.findall(r"erg_orb_l2_\d{8}_v\d{2}\.cdf", listing.text))
            for day in [day for day in missing if day.year == year]:
                candidates = sorted(name for name in names if name.startswith(f"erg_orb_l2_{day:%Y%m%d}_"))
                if not candidates:
                    raise FileNotFoundError(f"ORB not listed for {day:%Y-%m-%d}")
                name = candidates[-1]
                target = spedas_dir / "ergsc" / relative / name
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(".cdf.part")
                try:
                    with session.get(REMOTE_ROOT + relative + name, stream=True, timeout=(15, 90)) as response:
                        response.raise_for_status()
                        with partial.open("wb") as stream:
                            for chunk in response.iter_content(1024 * 1024):
                                stream.write(chunk)
                    read_orbit_day(partial, day)
                    partial.replace(target)
                finally:
                    partial.unlink(missing_ok=True)
                result[day] = target
                print(f"Downloaded ORB {day:%Y-%m-%d}: {target.stat().st_size} bytes", flush=True)
    return result


def read_orbit_day(path, day, gap_threshold_sec=30.0):
    with cdflib.CDF(path) as cdf:
        times = pd.DatetimeIndex(cdflib.cdfepoch.to_datetime(cdf.varget("epoch")))
        position = np.asarray(cdf.varget("pos_rmlatmlt"), dtype=float)
        l_values = np.asarray(cdf.varget("pos_Lm"), dtype=float)
    if position.shape != (len(times), 3) or l_values.shape != (len(times), 3):
        raise ValueError(f"Unexpected ORB dimensions: {path}")
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError(f"Non-increasing ORB times: {path}")
    end = day + pd.Timedelta(days=1)
    mask = (times >= day) & (times < end)
    if not mask.any():
        raise ValueError(f"ORB has no records for {day}: {path}")
    position[np.abs(position) >= 1e29] = np.nan
    l_values[np.abs(l_values) >= 1e29] = np.nan
    times = times[mask]
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    delta = np.diff(time_ns, append=end.value) / 1e9
    return pd.DataFrame({
        "time": times, "r_re": position[mask, 0], "mlat_deg": position[mask, 1],
        "mlt_hour": np.mod(position[mask, 2], 24.0), "l_90deg": l_values[mask, 0],
        "dwell_sec": np.where((delta > 0) & (delta <= gap_threshold_sec), delta, 0.),
    })
