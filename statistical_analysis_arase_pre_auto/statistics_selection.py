"""Post hoc occupied-block selection; stored analysis results remain unchanged."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import batch_arase as batch
from statistics_phases import PHASES


def minimum_blocks(value):
    value = int(value)
    if value < 6:
        raise ValueError("Saved fits require at least 6 blocks; a lower threshold needs reanalysis")
    return value


def add_selection_argument(parser):
    parser.add_argument("--min-occupied-blocks", type=minimum_blocks, default=10,
                        help="Minimum occupied 10-second blocks per accepted phase/range (default: 10).")


def selection_key(minimum):
    return f"min_blocks_{minimum_blocks(minimum)}"


def accepted_events(master, minimum):
    minimum = minimum_blocks(minimum)
    required = {"status", "n_occupied_blocks", "block_duration"}
    if not required.issubset(master.columns):
        raise KeyError(f"Master lacks occupied-block diagnostics: {sorted(required - set(master.columns))}")
    ok = master["status"].eq("ok")
    counts = pd.to_numeric(master["n_occupied_blocks"], errors="coerce")
    if (ok & (~np.isfinite(counts) | counts.lt(0) | counts.mod(1).ne(0))).any():
        raise ValueError("status=ok rows have invalid n_occupied_blocks")
    durations = pd.to_timedelta(master["block_duration"], errors="coerce")
    if (ok & durations.ne(pd.Timedelta(seconds=10))).any():
        raise ValueError("Occupied-block selection requires stored 10-second blocks")
    return ok & counts.ge(minimum)


def select_events(master, minimum):
    return master.loc[accepted_events(master, minimum)].copy()


def selection_metadata(master, minimum):
    accepted = accepted_events(master, minimum)
    return {
        "min_occupied_blocks": minimum_blocks(minimum), "block_duration": "10s",
        "condition": f"status == ok and n_occupied_blocks >= {minimum}",
        "phase_counts": {phase: {
            "before": int((master.phase_mode.eq(phase) & master.status.eq("ok")).sum()),
            "after": int((master.phase_mode.eq(phase) & accepted).sum()),
        } for phase in PHASES},
    }


def load_prepared_inputs(root, period):
    source = Path(root) / "inputs" / "provenance.json"
    provenance = json.loads(source.read_text())
    first = provenance.get("start_month") or f"{provenance['year']}01"
    last = provenance.get("end_month") or f"{provenance['year']}12"
    if (first != f"{period.start:%Y%m}" or last != f"{period.last_day:%Y%m}"
            or provenance["analysis_version"] != batch.ANALYSIS_VERSION):
        raise ValueError("Prepared statistics period/version mismatch")
    master_path = Path(root) / "inputs" / "arase_phase_master.csv"
    if batch.file_sha256(master_path) != provenance["annual_master_sha256"]:
        raise ValueError("Prepared master checksum mismatch")
    return pd.read_csv(master_path), provenance
