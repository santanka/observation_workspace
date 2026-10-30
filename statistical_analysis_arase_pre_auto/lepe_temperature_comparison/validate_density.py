"""Validate density artifacts and, optionally, auxiliary-field invariance on real CDF."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import density as dn


def validate(day_root, check_auxiliary=False, reference=False):
    folders = sorted(day_root.glob(f'*/{dn.core.OUTPUT_DIR}/{dn.VERSION}'))
    if not folders:
        raise ValueError('No density result folders')
    report = dict(events=len(folders), integral_records=0, raw_matched_in_event=0, raw_unmatched_in_event=0,
                  hfa_matched_event_minutes=0, hfa_valid_event_minutes=0, max_density_partition_difference_cm3=0.)
    input_checked, minute_tables = set(), []
    for folder in folders:
        meta = json.loads((folder / dn.PROVENANCE).read_text())
        if meta['code_sha256'] != dn.code_hashes():
            raise ValueError('Saved numerical code differs from current code')
        for name, digest in meta['output_sha256'].items():
            if dn.core.sha256(folder / name) != digest:
                raise ValueError('Output checksum mismatch: ' + str(folder / name))
        for source in meta['inputs'].values():
            if source['path'] not in input_checked:
                if dn.core.sha256(source['path']) != source['sha256']:
                    raise ValueError('Input checksum mismatch: ' + source['path'])
                input_checked.add(source['path'])
        frame = pd.read_csv(folder / 'native_comparison.csv', float_precision='round_trip')
        n = frame[frame.record_has_integral]
        report['integral_records'] += len(n)
        all_density = n.all__density_raw_cm3.to_numpy()
        split = n.low__density_raw_cm3.to_numpy() + n.high__density_raw_cm3.to_numpy()
        centers = meta['integral_profiles']
        assert centers['low']['actual_energy_centers_eV'][-1] < centers['high']['actual_energy_centers_eV'][0]
        np.testing.assert_allclose(all_density, split, rtol=2e-12, equal_nan=True)
        report['max_density_partition_difference_cm3'] = max(
            report['max_density_partition_difference_cm3'], float(np.nanmax(abs(all_density - split))))
        if reference:
            old_path = Path(meta['original_event_folder']) / dn.core.OUTPUT_DIR / dn.mb.VERSION / 'native_comparison.csv'
            old = pd.read_csv(old_path, float_precision='round_trip')
            old = old[old.record_has_integral]
            for band in dn.mb.BANDS:
                np.testing.assert_allclose(n[f'{band}__density_raw_cm3'], old[f'{band}__density_cm3'], rtol=1e-12, equal_nan=True)
        matched = n['raw_match__status'] == 'matched'
        assert n.loc[matched, 'raw_match__pi_source_index'].is_unique
        assert n.loc[matched, 'raw_match__nearest_offset_s'].abs().max() <= meta['options']['raw_match_tolerance_ms'] / 1000
        with np.load(folder / 'diagnostics.npz', allow_pickle=False) as d:
            for band in dn.mb.BANDS:
                np.testing.assert_allclose(d[f'pi_{band}__density_raw_cm3'], d[f'pi_{band}__ne'].astype('float64') * 1e-6,
                                           rtol=1e-15, equal_nan=True)
            good = d['hfa__valid']
            np.testing.assert_array_equal(d['hfa__quality_raw'][good], 0)
            assert np.issubdtype(d['hfa__quality_raw'].dtype, np.integer)
        table = pd.read_csv(folder / 'minute_comparison.csv', float_precision='round_trip')
        index = pd.DatetimeIndex(pd.to_datetime(table.pop('time_utc')), name='time_utc')
        table.index = index
        matched_hfa = table.hfa_status == 'matched'
        saved = pd.to_datetime(table.loc[matched_hfa, 'hfa_time_utc']).to_numpy(dtype='datetime64[ns]').astype('int64')
        nominal = (index[matched_hfa] + pd.Timedelta(seconds=30)).to_numpy(dtype='datetime64[ns]').astype('int64')
        np.testing.assert_allclose(table.loc[matched_hfa, 'hfa_nearest_offset_s'], (saved - nominal) / 1e9, atol=1e-12)
        assert table.loc[matched_hfa, 'hfa_nearest_offset_s'].abs().max() <= meta['options']['hfa_match_tolerance_s']
        for band in dn.mb.BANDS:
            for key in ('integral', 'fit'):
                expected = table[f'{band}__{key}_density_cm3'] / table.hfa_density_cm3
                np.testing.assert_allclose(table[f'{band}__{key}_over_hfa'], expected, rtol=1e-14, equal_nan=True)
        minute_tables.append(table)
        summary = meta['summary']
        for out, name in [('raw_matched_in_event', 'raw_n_matched_in_event'), ('raw_unmatched_in_event', 'raw_n_unmatched_in_event'),
                          ('hfa_matched_event_minutes', 'hfa_n_time_matched_minutes'), ('hfa_valid_event_minutes', 'hfa_n_valid_minutes')]:
            report[out] += summary[name]
    pooled = dn.pooled_minute_summary(minute_tables)
    report['hfa_valid_unique_minutes'] = int(pooled.n_hfa_valid_minutes.iloc[0])
    report['unique_minutes'] = int(pooled.n_unique_minutes.iloc[0])
    if check_auxiliary:
        first = folders[0]
        meta = json.loads((first / dn.PROVENANCE).read_text())
        old = pd.read_csv(first / 'native_comparison.csv', float_precision='round_trip')
        old = old[old.record_has_integral]
        lo, hi = pd.Timestamp(meta['event_start_utc']).floor('min'), pd.Timestamp(meta['event_end_utc']).ceil('min')
        from cdflib import CDF
        maximum = 0.
        with dn.core.staged_cdf(meta['inputs']['lepe']['path'], Path('/tmp')) as path:
            cdf = CDF(path)
            for band in dn.mb.BANDS:
                mask = int(meta['pi_profiles'][band]['attributes']['packet_info_bitmask'])
                other, _ = dn.compute_density(cdf, path, lo, hi, band, mask, auxiliary_field=(0., 1., 0.))
                np.testing.assert_allclose(other.density_raw_cm3, old[f'{band}__density_raw_cm3'], rtol=1e-12, equal_nan=True)
                maximum = max(maximum, float(np.nanmax(abs(other.density_raw_cm3.to_numpy() - old[f'{band}__density_raw_cm3'].to_numpy()))))
        report['auxiliary_field_max_density_difference_cm3'] = maximum
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day-root', type=Path, required=True)
    parser.add_argument('--check-auxiliary-field', action='store_true')
    parser.add_argument('--reference-temperature-v3', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = validate(args.day_root, args.check_auxiliary_field, args.reference_temperature_v3)
    text = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
