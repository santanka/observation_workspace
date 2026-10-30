"""Map geographic traveling directions to hemisphere-relative directions.

Use the range-mean signed magnetic latitude (degrees), not fit-only latitude.
Retain the original phase and all measurements; ambiguous rows remain recorded.
"""

import numpy as np
import pandas as pd


SOURCE_PHASES = ("all", "north_traveling", "south_traveling", "standing")
PHASES = {
    "all": "All phase components",
    "toward_ionosphere": "Toward ionosphere",
    "toward_magnetic_equator": "Toward magnetic equator",
    "standing": "Standing",
}


def classify_phases(master):
    frame = master.copy()
    if "source_phase_mode" not in frame:
        frame["source_phase_mode"] = frame["phase_mode"]
    source = frame["source_phase_mode"]
    if not source.isin(SOURCE_PHASES).all():
        raise ValueError("Unknown source_phase_mode in statistics input")
    frame["phase_mode"] = source
    frame["direction_classification_reason"] = "unchanged"
    traveling = source.isin(["north_traveling", "south_traveling"])
    if not traveling.any():
        return frame
    latitude = pd.to_numeric(frame["range_MLAT_deg_mean"], errors="coerce")
    finite = np.isfinite(latitude)
    valid = traveling & finite & latitude.ne(0)
    ionosphere = ((source.eq("north_traveling") & latitude.gt(0))
                  | (source.eq("south_traveling") & latitude.lt(0)))
    frame.loc[traveling, "phase_mode"] = "unclassified"
    frame.loc[traveling & ~finite, "direction_classification_reason"] = "missing_or_nonfinite_mlat"
    frame.loc[traveling & finite & latitude.eq(0), "direction_classification_reason"] = "zero_mlat"
    frame.loc[valid & ionosphere, "phase_mode"] = "toward_ionosphere"
    frame.loc[valid & ~ionosphere, "phase_mode"] = "toward_magnetic_equator"
    frame.loc[valid, "direction_classification_reason"] = "range_mean_mlat_sign"
    return frame
