"""Validate saved v3 checksums, raw matching and independent moment identities."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare as core
import multiband as mb


def validate(day_root, reference=False):
    folders = sorted(day_root.glob(f'*/{core.OUTPUT_DIR}/{mb.VERSION}'))
    if not folders:
        raise ValueError('No saved v3 event folders')
    result = dict(events=len(folders), native_raw_matched_in_event=0, native_raw_unmatched_in_event=0,
                  tensor_partition_n_checked=0, max_partition_tensor_difference_eV=0.)
    for folder in folders:
        meta = json.loads((folder / mb.PROVENANCE).read_text())
        for name, digest in meta['output_sha256'].items():
            if core.sha256(folder / name) != digest:
                raise ValueError(f'Output checksum mismatch: {folder / name}')
        if meta['code_sha256'] != mb.code_hashes():
            raise ValueError(f'Saved numerical code differs from current code: {folder}')
        for key in ('native_raw_matched_in_event', 'native_raw_unmatched_in_event'):
            result[key] += meta['summary'][key.replace('native_raw_', 'raw_n_')]
        frame = pd.read_csv(folder / 'native_comparison.csv')
        n = frame[frame.record_has_integral]
        matched = n['raw_match__status'] == 'matched'
        if matched.any():
            assert n.loc[matched, 'raw_match__nearest_offset_s'].abs().max() <= meta['options']['raw_match_tolerance_ms'] / 1000
            assert n.loc[matched, 'raw_match__pi_source_index'].is_unique
        density = {k: n[f'{k}__density_cm3'].to_numpy() for k in mb.BANDS}
        bounds = {k: v['integral_actual_centers_bounds_eV'] for k, v in meta['profiles'].items()}
        split_covers_full = bounds['low'][0] == bounds['all'][0] and bounds['low'][1] < bounds['high'][0] and bounds['high'][1] == bounds['all'][1]
        if not split_covers_full:
            raise ValueError('Density/tensor partition check requires disjoint low/high channels covering full range')
        np.testing.assert_allclose(density['all'], density['low'] + density['high'], rtol=2e-12, equal_nan=True)
        with np.load(folder / 'diagnostics.npz', allow_pickle=False) as d:
            valid = np.ones(len(n), dtype=bool)
            for band in mb.BANDS:
                valid &= (np.isfinite(density[band]) & (density[band] > 0)
                          & np.isfinite(d[band + '__ttens_dsi_eV']).all(axis=(1, 2))
                          & np.isfinite(d[band + '__velocity_dsi_kms']).all(axis=1))
                tensor = d[band + '__ttens_dsi_eV']
                np.testing.assert_allclose(np.trace(tensor, axis1=1, axis2=2) / 3,
                                           d[band + '__raw_temperature_eV'][:, 0],
                                           rtol=1e-12, atol=1e-8, equal_nan=True)
            if valid.any():
                total = np.zeros_like(d['all__ttens_dsi_eV'][valid])
                for band in ('low', 'high'):
                    velocity_delta = d[band + '__velocity_dsi_kms'][valid] - d['all__velocity_dsi_kms'][valid]
                    # Public LEP-e moment electron mass in eV / (km/s)^2.
                    drift = 5.68566e-6 * np.einsum('ni,nj->nij', velocity_delta, velocity_delta)
                    total += density[band][valid, None, None] * (d[band + '__ttens_dsi_eV'][valid] + drift)
                total /= density['all'][valid, None, None]
                np.testing.assert_allclose(total, d['all__ttens_dsi_eV'][valid], rtol=3e-10, atol=1e-8)
                result['tensor_partition_n_checked'] += int(valid.sum())
                result['max_partition_tensor_difference_eV'] = max(
                    result['max_partition_tensor_difference_eV'],
                    float(np.max(abs(total - d['all__ttens_dsi_eV'][valid]))))
        if reference:
            old = pd.read_csv(Path(meta['original_event_folder']) / core.OUTPUT_DIR / 'lepe_temperature_comparison_dual_energy_1min.csv')
            new = pd.read_csv(folder / 'minute_comparison.csv')
            for key in core.COMPONENTS:
                np.testing.assert_allclose(new[f'low__moment_{key}_eV'], old[f'moment_le1000_{key}_eV'],
                                           rtol=1e-12, equal_nan=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day-root', type=Path, required=True)
    parser.add_argument('--reference-dual-energy', action='store_true', help='Compare low Integral against existing dual-energy outputs')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = validate(args.day_root, args.reference_dual_energy)
    content = json.dumps(report, indent=2) + '\n'
    if args.report:
        args.report.write_text(content)
    print(content, end='')


if __name__ == '__main__':
    main()
