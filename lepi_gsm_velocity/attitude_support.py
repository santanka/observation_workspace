"""Strict fallback reader for adjacent signed ATT fields, without rewriting raw TXT."""
import datetime as dt
from pathlib import Path
import re

import numpy as np
import convert as c

NUMBER = re.compile(r'[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf(?:inity)?)', re.I)


def numeric_tokens(token):
    """Split 26.2539-151515.3574, retaining exponent signs and rejecting junk."""
    matches = list(NUMBER.finditer(token))
    if not matches or ''.join(m.group() for m in matches) != token:
        return [token]
    if any(m.group()[0] not in '+-' for m in matches[1:]):
        raise ValueError('Ambiguous adjacent attitude numbers: ' + token)
    return [m.group() for m in matches]


def parse_attitude(content):
    if 'J2000' not in content:
        raise ValueError('Attitude file does not declare J2000 axes')
    times, angles, quality = [], [], []
    repaired = 0
    for line_number, line in enumerate(content.splitlines(), 1):
        raw = line.split()
        if not raw or raw[0].startswith('#'):
            continue
        fields = [raw[0]]
        for token in raw[1:]:
            fields.extend(numeric_tokens(token))
        if len(fields) == 16:
            numeric = fields[1:7] + fields[9:16]
            flags, age = fields[7:9], fields[14:16]
        elif len(fields) == 14:
            numeric = fields[1:14]
            flags, age = ['', ''], fields[12:14]
        else:
            raise ValueError(f'Unsupported attitude layout at line {line_number}: {len(fields)} fields')
        try:
            [float(value) for value in numeric]
        except ValueError as exc:
            raise ValueError(f'Invalid numeric attitude field at line {line_number}') from exc
        times.append(fields[0]); angles.append([float(fields[2]), float(fields[3])])
        quality.append(flags + age)
        repaired += fields != raw
    if not times:
        raise ValueError('Empty attitude ASCII data')
    return times, np.asarray(angles, dtype=float), np.asarray(quality), repaired


class Support(c.Support):
    """Retain the legacy path/signature for successful days; fallback only on parse errors."""
    def load(self, day):
        try:
            return super().load(day)
        except RuntimeError as exc:
            reason = str(exc)
            if not any(text in reason for text in ('could not convert string to float', 'Unsupported attitude row layout')):
                raise
        import pyspedas as psp
        from pyspedas.projects.erg.satellite.erg.att.att import att
        from pyspedas.projects.erg.satellite.erg.orb.orb import orb
        from pyspedas.tplot_tools import time_float
        import cdflib
        trange = [(day - dt.timedelta(seconds=120)).strftime('%Y-%m-%d/%H:%M:%S'),
                  (day + dt.timedelta(days=1, seconds=120)).strftime('%Y-%m-%d/%H:%M:%S')]
        # This fallback is deliberately local-only, including when called independently.
        af = att(trange=trange, downloadonly=True, no_update=True, force_download=False)
        of = orb(trange=trange, level='l2', datatype='def', downloadonly=True,
                 no_update=True, force_download=False, ror=False)
        if not af or not of:
            raise RuntimeError('Local attitude/orbit support missing during ATT fallback')
        inputs = [{'fingerprint': c.fp(p), 'sha256': c.sha(p),
                   'role': 'attitude_ascii' if str(p).endswith('.txt') else 'orbit_cdf'}
                  for p in sorted(set(af + of))]
        raw_times, raw_angles, raw_quality = [], [], []
        repaired = 0
        for name in sorted(set(af)):
            times, angles, quality, count = parse_attitude(Path(name).read_text())
            raw_times.extend(times); raw_angles.extend(angles); raw_quality.extend(quality)
            repaired += count
        times = np.asarray(time_float(raw_times))
        angles, quality = np.asarray(raw_angles), np.asarray(raw_quality)
        order = np.argsort(times, kind='stable')
        at, combined = c.series(times[order], np.column_stack((angles.astype(str), quality))[order])
        angles, quality = combined[:, :2].astype(float), combined[:, 2:]
        psp.del_data('erg_orb_l2_*')
        orb(trange=trange, level='l2', datatype='def', no_update=True, ror=False)
        pos = psp.get_data('erg_orb_l2_pos_gse')
        if pos is None:
            raise RuntimeError('Required orbit variable missing after ATT fallback')
        ot, positions = c.series(pos.times, pos.y)
        units = set()
        for name in of:
            with cdflib.CDF(name) as cdf:
                unit = str(cdf.varattsget('pos_gse').get('UNITS', '')).strip().lower()
                if unit not in ('km', 're'):
                    raise ValueError('Unsupported orbit units: ' + unit)
                units.add(unit)
                cdf.varget('pos_gse')
        if len(units) != 1:
            raise ValueError('Mixed orbit units in ATT fallback')
        if units == {'re'}:
            positions = positions * self.args.earth_radius_km
        code = Path(__file__).resolve()
        inputs.append({'fingerprint': c.fp(code), 'sha256': c.sha(code),
                       'role': 'attitude_reader_code:adjacent_signed_numbers_v1',
                       'repaired_attitude_rows_in_loaded_files': repaired,
                       'legacy_failure': reason})
        c.assert_snapshot(inputs)
        self.day, self.at, self.angles = day, at, angles
        self.ot, self.positions, self.quality = ot, positions, quality
        self.inputs, self.orbit_input_units = inputs, next(iter(units))
