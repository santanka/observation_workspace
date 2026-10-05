"""Tolerant local loader for Arase ATT L2 text files.

Some official files omit the delimiter between GZ-Delta and a negative
ADSG_interp_dt value (for example ``26.2539-151515.3574``).  The upstream
PySPEDAS loader splits only on whitespace, so one malformed auxiliary column
prevents every range on that day from loading.  This module preserves the
upstream column interpretation and repairs only that missing delimiter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pyspedas.tplot_tools import store_data, time_float


# The affected boundary is between two fixed 4-decimal numeric fields.  Keep
# the lookbehind specific so ISO dates (YYYY-MM-DD) and exponent notation are
# never altered.
_MISSING_DELIMITER = re.compile(r"(?<=\d\.\d{4})(?=-\d+\.\d{4}(?:\s|$))")
_TPLOT_COLUMNS = {
    "erg_att_sprate": 1,
    "erg_att_spphase": 9,
    "erg_att_izras": 2,
    "erg_att_izdec": 3,
    "erg_att_gxras": 10,
    "erg_att_gxdec": 11,
    "erg_att_gzras": 12,
    "erg_att_gzdec": 13,
}


def repair_missing_att_delimiter(value: str) -> str:
    """Insert whitespace before an attached negative auxiliary value."""
    return _MISSING_DELIMITER.sub(" ", value)


def _read_att_frames(files: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in files:
        raw = pd.read_table(file)
        series = raw.iloc[10:, 0].astype("string")
        repaired = series.map(repair_missing_att_delimiter)
        frames.append(repaired.str.split(expand=True))
    if not frames:
        raise ValueError("No local ATT files were supplied")
    frame = pd.concat(frames, ignore_index=True)
    if frame.shape[1] < 14:
        raise ValueError(f"ATT table has only {frame.shape[1]} columns; expected at least 14")
    return frame


def parse_att_files(files: Iterable[str | Path], *, store: bool = True):
    """Parse local ATT files with the upstream PySPEDAS column convention."""
    frame = _read_att_frames(files)
    times = time_float(frame.iloc[:, 0])
    # Match the upstream loader: official "NaN" tokens are missing values,
    # while other nonnumeric strings still raise instead of being coerced.
    values = {
        name: frame.iloc[:, column].astype(float).to_numpy()
        for name, column in _TPLOT_COLUMNS.items()
    }
    if store:
        for name, value in values.items():
            store_data(name, data={"x": times, "y": value})
        return list(_TPLOT_COLUMNS)
    return {
        name: {"x": times, "y": value}
        for name, value in values.items()
    }


def validate_att_files(files: Iterable[str | Path]) -> None:
    """Raise when an ATT file cannot be parsed into the required columns."""
    parse_att_files(files, store=False)


def load_att_tolerant(
    trange,
    *,
    level: str = "l2",
    no_update: bool = True,
    downloadonly: bool = False,
    notplot: bool = False,
):
    """Resolve local files through PySPEDAS, then parse them tolerantly."""
    import pyspedas as psp

    files = psp.projects.erg.att(
        trange=trange,
        level=level,
        no_update=no_update,
        downloadonly=True,
    )
    files = [str(path) for path in files or [] if Path(path).is_file()]
    if not files:
        raise FileNotFoundError(f"No local ATT files for {trange}")
    if downloadonly:
        return files
    return parse_att_files(files, store=not notplot)
