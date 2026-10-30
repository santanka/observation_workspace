"""MGF record quarantine shared by validation and computation; never modifies CDFs."""
import re
from pathlib import Path
import numpy as np

POLICY = 'mgf_record_quarantine_tt2000_v1'


def nominal_time(source, cadence):
    if 'nominal_utc' in source:  # Explicit nominal granule for synthetic fixtures.
        return np.datetime64(source['nominal_utc'], 'ns')
    name = Path(source['source']['path']).name
    pattern = r'_(\d{10})_v' if cadence == '64hz' else r'_(\d{8})_v'
    match = re.search(pattern, name)
    if not match:
        raise ValueError('Cannot establish nominal MGF granule: ' + name)
    stamp = match.group(1)
    return np.datetime64(f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}' +
                         (f'T{stamp[8:10]}:00:00' if cadence == '64hz' else 'T00:00:00'), 'ns')


def record_selection(epoch, dates, nominal, cadence, tolerance=32.):
    duration = np.timedelta64(1, 'h' if cadence == '64hz' else 'D')
    margin = np.timedelta64(int(tolerance * 1e9), 'ns')
    outside = np.isnat(dates) | (dates < nominal - margin) | (dates > nominal + duration + margin)
    keep = np.flatnonzero(~outside)
    if np.any(np.diff(epoch[keep]) < 0):
        raise ValueError('MGF epoch reversal remains after out-of-granule quarantine')
    # Distinct segments prevent interpolation across removed interior records.
    segments = np.cumsum(outside, dtype=np.int64)
    return keep, outside, segments


def conflicting_pairs(epoch, payload):
    duplicate = np.flatnonzero(np.diff(epoch) == 0)
    if not len(duplicate):
        return duplicate
    a, b = payload[duplicate], payload[duplicate + 1]
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    same = same.reshape(len(duplicate), -1).all(axis=1)
    return duplicate[~same]
