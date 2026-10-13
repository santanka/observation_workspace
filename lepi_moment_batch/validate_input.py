"""Isolated, bounded CDF integrity/schema validator. No tplot or moment calculation."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import resource

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
import numpy as np
from cdflib import CDF, cdfepoch


class DataPolicyError(ValueError):
    def __init__(self, message, **diagnostics):
        self.diagnostics = diagnostics
        super().__init__(message)


def validate(path, product, date, boundary_tolerance_seconds=None):
    if boundary_tolerance_seconds is None:
        boundary_tolerance_seconds = 64. if product == 'lepi' else 32.
    margin = np.timedelta64(int(boundary_tolerance_seconds * 1e9), 'ns')
    path = Path(path)
    with path.open('rb') as f:
        if f.read(8) != bytes.fromhex('cdf300010000ffff'):
            raise ValueError('Expected expanded CDF3')
        f.seek(20)
        gdr = int.from_bytes(f.read(8), 'big')
        f.seek(gdr + 36)
        logical_eof = int.from_bytes(f.read(8), 'big')
    cdf = CDF(path)
    if path.stat().st_size != logical_eof + (16 if cdf._md5 else 0):
        raise ValueError('CDF logical EOF differs from file length')
    if cdf._md5 and not cdf._md5_validation():
        raise ValueError('CDF embedded checksum failed')
    epoch_name = 'Epoch' if product == 'lepi' else 'epoch_' + product
    epoch = np.asarray(cdf.varget(epoch_name))
    if cdf.varinq(epoch_name).Data_Type != 33 or epoch.ndim != 1 or not len(epoch):
        raise ValueError('Expected nonempty scalar TT2000 epochs')
    if np.any(np.diff(epoch) < 0):
        raise DataPolicyError('Epoch reversal requires inspection; not a transfer-corruption test',
                              reverse_pairs=int((np.diff(epoch) < 0).sum()))
    duplicates = np.flatnonzero(np.diff(epoch) == 0)
    dates = cdfepoch.to_datetime(epoch)
    nominal = np.datetime64(date, 'ns')
    duration = np.timedelta64(1, 'h' if product == '64hz' else 'D')
    if (np.isnat(dates).any() or dates[0] < nominal - margin
            or dates[-1] > nominal + duration + margin):
        outside = np.isnat(dates) | (dates < nominal - margin) | (dates > nominal + duration + margin)
        raise DataPolicyError(f'Epochs outside nominal granule ({boundary_tolerance_seconds:g}-second boundary tolerance)',
                              first_epoch_utc=str(dates[0]), last_epoch_utc=str(dates[-1]),
                              outside_records=int(outside.sum()),
                              outside_examples=[{'record': int(i), 'utc': str(dates[i])}
                                                for i in np.flatnonzero(outside)[:10]])
    n = len(epoch)
    required = {'epoch_sp': []} if product == 'lepi' else {'mag_' + product + '_dsi': [3],
                                                          'quality_' + product: None}
    if product == 'lepi':
        for species in ('FPDU', 'FHEDU', 'FODU'):
            required.update({species: [32, 8, 16], species + '_COUNT_RAW': [32, 8, 16],
                             species + '_Quality': [32, 15, 16]})
            energy_name = species + '_Energy'
            if cdf.varinq(energy_name).Rec_Vary:
                raise ValueError('Time-dependent energy table unsupported')
            attributes = cdf.varattsget(species)
            if attributes.get('DEPEND_0') != 'Epoch' or attributes.get('DEPEND_1') != energy_name:
                raise ValueError('Missing/inconsistent flux dependencies: ' + species)
            if not attributes.get('UNITS'):
                raise ValueError('Missing flux units: ' + species)
            energy = np.asarray(cdf.varget(energy_name))
            if energy.shape != (32,) or not np.all(np.isfinite(energy[:30]) & (energy[:30] > 0)):
                raise ValueError('Invalid energy table: ' + species)
    # Read every required record with small buffers. NaN/fill is scientific quality, not corruption.
    checks = {}
    for name, dimensions in required.items():
        q = cdf.varinq(name)
        if not q.Rec_Vary or q.Last_Rec + 1 != n or (dimensions is not None and q.Dim_Sizes != dimensions):
            raise ValueError('Required variable shape/record mismatch: ' + name)
        for first in range(0, n, 1024):
            last = min(first + 1023, n - 1)
            data = np.asarray(cdf.varget(name, startrec=first, endrec=last))
            if data.shape != (last - first + 1, *q.Dim_Sizes):
                raise ValueError('Incomplete required records: ' + name)
        # A repeated timestamp is safe to collapse only when every required payload is identical.
        for index in duplicates:
            pair = np.asarray(cdf.varget(name, startrec=int(index), endrec=int(index + 1)))
            if not np.array_equal(pair[0], pair[1], equal_nan=True):
                raise DataPolicyError('Conflicting records at a duplicate epoch: ' + name,
                                      duplicate_records=[int(index), int(index + 1)],
                                      duplicate_epoch_utc=str(dates[index]))
        checks[name] = {'records': n, 'dimensions': q.Dim_Sizes}
    return {'status': 'verified', 'product': product, 'records': n,
            'first_epoch_utc': str(dates[0]), 'last_epoch_utc': str(dates[-1]),
            'identical_duplicate_pairs': int(len(duplicates)),
            'record_policy': 'nondecreasing epochs; identical duplicates only; retain last original record',
            'boundary_tolerance_seconds': boundary_tolerance_seconds,
            'logical_eof_bytes': logical_eof, 'required_variables': checks,
            'full_required_payload_read': True,
            'coverage_note': 'Internal gaps/partial granule coverage allowed; no quality selection'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    parser.add_argument('--product', choices=['lepi', '64hz', '8sec'], required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--boundary-tolerance-seconds', type=float)
    args = parser.parse_args()
    try:
        report = validate(args.path, args.product, args.date, args.boundary_tolerance_seconds)
    except Exception as exc:
        report = {'status': 'failed_validation', 'reason': str(exc), 'exception': type(exc).__name__,
                  'retryable': not isinstance(exc, DataPolicyError),
                  **getattr(exc, 'diagnostics', {})}
    Path(args.result).write_text(json.dumps(report, indent=2))
