"""Known-truth density units, quality, alignment, pooling and restart checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import density as dn


def pi_frame(index):
    return pd.DataFrame(dict(ne=np.full(len(index), 1e6), T_para=np.full(len(index), 100.),
                              T_perp=np.full(len(index), 200.), num_valid_data=np.full(len(index), 200),
                              log_residual=np.full(len(index), .5), eflux_para_mean=np.full(len(index), 2e4),
                              eflux_perp_mean=np.full(len(index), 2e4)), index=index)


def integral_frame(index):
    return pd.DataFrame(dict(density_raw_cm3=np.ones(len(index)), packet_info_raw=np.zeros(len(index), dtype=int),
                              n_finite_flux=np.full(len(index), 100), n_selected_flux=np.full(len(index), 100),
                              n_negative_flux=np.zeros(len(index))), index=index)


class DensityTests(unittest.TestCase):
    def test_pi_units_and_independent_fitted_density_mask(self):
        index = pd.date_range('2022-09-01', periods=4, freq='1min')
        f = pi_frame(index)
        f.loc[index[1], 'T_para'] = np.nan
        f.loc[index[2], 'ne'] = -1
        f.loc[index[3], 'num_valid_data'] = 3
        out = dn.pi_density(f)
        self.assertEqual(out.density_cm3.iloc[0], 1.)
        self.assertEqual(out['ne'].iloc[0], 1e6)
        np.testing.assert_array_equal(out.reason, [0, 2, 1, 4])
        self.assertTrue(out.density_cm3.iloc[1:].isna().all())

    def test_pi_density_requires_both_flux_directions_only_if_configured(self):
        index = pd.date_range('2022-09-01', periods=2, freq='1min')
        f = pi_frame(index)
        f.loc[index[0], 'eflux_perp_mean'] = 1e4
        f.loc[index[1], 'eflux_para_mean'] = -1
        low, high = dn.pi_density(f, 1e4), dn.pi_density(f)
        self.assertFalse(low.valid.any())
        self.assertTrue(high.valid.all())
        self.assertTrue(high.flux_warning.iloc[1])
        np.testing.assert_array_equal(low.reason, [8, 8])

    def test_integral_ignores_temperature_magnetic_masks_and_rejects_packet(self):
        index = pd.date_range('2022-09-01', periods=4, freq='8s')
        f = integral_frame(index)
        f['magnetic_valid'] = False
        f['valid'] = False  # A temperature mask from another workflow must be ignored.
        f['temperature'] = np.nan
        f.loc[index[1], 'packet_info_raw'] = 2
        f.loc[index[2], 'n_finite_flux'] = 0
        f.loc[index[3], 'density_raw_cm3'] = 0
        out = dn.integral_quality(f, 2)
        np.testing.assert_array_equal(out.reason, [0, 2, 4, 1])
        self.assertEqual(out.density_cm3.iloc[0], 1.)

    def test_hfa_exact_zero_flags_including_large_bitfield_and_negative_fill(self):
        f = pd.DataFrame(dict(density_raw_cm3=[1., 2., 3., 4., -1., 1., np.nan],
                              quality_raw=[0, 2, 262144, -128, 0, np.nan, 0]))
        out = dn.hfa_quality(f)
        np.testing.assert_array_equal(out.reason, [0, 2, 2, 2, 1, 2, 1])
        self.assertEqual(int(out.valid.sum()), 1)

    def test_hfa_csv_round_trip_preserves_ratio_precision(self):
        hfa = dn.hfa_quality(pd.DataFrame(dict(density_raw_cm3=np.array([3.1415927], dtype='float32'),
                                                quality_raw=np.array([0], dtype='uint32'))))
        table = hfa[['density_cm3']].copy()
        table['integral_density_cm3'] = .12345678912345678
        table['ratio'] = table.integral_density_cm3 / table.density_cm3
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'table.csv'
            table.to_csv(path, index=False)
            restored = pd.read_csv(path, float_precision='round_trip')
        np.testing.assert_allclose(restored.ratio, restored.integral_density_cm3 / restored.density_cm3, rtol=1e-14)
        self.assertEqual(hfa.density_raw_cm3.dtype, np.dtype('float32'))

    def test_hfa_center_alignment_preserves_actual_epochs_and_bad_quality(self):
        bins = pd.date_range('2022-09-01', periods=3, freq='1min', name='time_utc')
        t = pd.DatetimeIndex([bins[0] + pd.Timedelta(seconds=29.998),
                              bins[1] + pd.Timedelta(seconds=30.003)])
        hfa = dn.hfa_quality(pd.DataFrame(dict(density_raw_cm3=[2., 4.], quality_raw=[0, 2]), index=t))
        out = dn.align_hfa(bins, hfa)
        np.testing.assert_allclose(out.hfa_nearest_offset_s.iloc[:2], [-.002, .003])
        self.assertEqual(out.hfa_time_utc.iloc[0], t[0])
        self.assertEqual(out.hfa_density_cm3.iloc[0], 2.)
        self.assertTrue(np.isnan(out.hfa_density_cm3.iloc[1]))
        self.assertEqual(out.hfa_status.iloc[2], 'outside_tolerance')
        self.assertTrue(np.isnan(out.hfa_density_cm3.iloc[2]))

    def test_hfa_tolerance_inclusive_boundary_and_no_gap_filling(self):
        bins = pd.date_range('2022-09-01', periods=3, freq='1min', name='time_utc')
        t = pd.DatetimeIndex([bins[0] + pd.Timedelta(seconds=31),
                              bins[2] + pd.Timedelta(seconds=31, microseconds=1)])
        hfa = dn.hfa_quality(pd.DataFrame(dict(density_raw_cm3=[2., 4.], quality_raw=[0, 0]), index=t))
        out = dn.align_hfa(bins, hfa)
        self.assertEqual(out.hfa_status.tolist(), ['matched', 'outside_tolerance', 'outside_tolerance'])

    def test_hfa_competing_matches_are_not_reused(self):
        t = pd.Timestamp('2022-09-01T00:00:30')
        hfa = dn.hfa_quality(pd.DataFrame(dict(density_raw_cm3=[2.], quality_raw=[0]), index=[t]))
        bins = pd.DatetimeIndex([t - pd.Timedelta(seconds=30), t - pd.Timedelta(seconds=29.5)])
        out = dn.align_hfa(bins, hfa)
        self.assertEqual(out.hfa_status.tolist(), ['ambiguous', 'ambiguous'])

    def test_minute_raw_density_means_and_hfa_ratios_with_missing_reference(self):
        bins = pd.date_range('2022-09-01', periods=2, freq='1min', name='time_utc')
        native_index = pd.to_datetime(['2022-09-01T00:00:59', '2022-09-01T00:01:00'])
        n = dn.integral_quality(integral_frame(native_index), 2)
        n['density_cm3'] = [2., 4.]
        fits = {band: dn.pi_density(pi_frame(native_index if band == 'all' else bins)) for band in dn.mb.BANDS}
        fits['all']['density_cm3'] = [6., 8.]
        hfa = dn.hfa_quality(pd.DataFrame(dict(density_raw_cm3=[2., 4.], quality_raw=[0, 2]),
                                          index=bins + pd.Timedelta(seconds=30)))
        table = dn.minute_table({band: n for band in dn.mb.BANDS}, fits, hfa, bins[0],
                                 bins[-1] + pd.Timedelta(minutes=1), 1.)
        np.testing.assert_allclose(table.all__integral_density_cm3, [2., 4.])
        np.testing.assert_allclose(table.all__fit_density_cm3, [6., 8.])
        self.assertEqual(table.all__integral_over_hfa.iloc[0], 1.)
        self.assertEqual(table.all__fit_over_hfa.iloc[0], 3.)
        self.assertTrue(np.isnan(table.all__fit_over_hfa.iloc[1]))
        self.assertEqual(table.all__fit_density_cm3.iloc[1], 8.)

    def test_pool_counts_overlapping_event_minutes_once(self):
        bins = pd.date_range('2022-09-01', periods=2, freq='1min')
        table = pd.DataFrame({'hfa_density_cm3': [2., 4.]}, index=bins)
        for band in dn.mb.BANDS:
            for key in ('integral', 'fit'):
                table[f'{band}__{key}_over_hfa'] = [1., 3.]
        summary = dn.pooled_minute_summary([table, table.iloc[1:]])
        self.assertTrue(summary.n_unique_minutes.eq(2).all())
        self.assertTrue(summary.integral_over_hfa_median.eq(2.).all())
        changed = table.iloc[1:].copy()
        changed['all__fit_over_hfa'] = 100.
        with self.assertRaises(AssertionError):
            dn.pooled_minute_summary([table, changed])

    def test_output_corruption_and_changed_hfa_signature_prevent_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in dn.OUTPUTS:
                (folder / name).write_bytes(b'output')
            meta = dict(signature='with_hfa_hash', summary={'event': 'event'},
                        output_sha256={name: dn.core.sha256(folder / name) for name in dn.OUTPUTS})
            (folder / dn.PROVENANCE).write_text(json.dumps(meta))
            self.assertEqual(dn.reusable(folder, 'with_hfa_hash'), meta['summary'])
            self.assertIsNone(dn.reusable(folder, 'changed_hfa_hash'))
            (folder / 'minute_comparison.csv').write_bytes(b'corrupted')
            self.assertIsNone(dn.reusable(folder, 'with_hfa_hash'))

    def test_dry_run_never_computes_density(self):
        args = SimpleNamespace(dataset=Path('/unused'), date='2022-09-01', dry_run=True)
        with patch.object(dn.core, 'discover_events', return_value=[]), patch.object(dn.core, 'check_events'), \
             patch.object(dn, 'input_files', return_value={}), patch.object(dn, 'compute_density') as compute:
            self.assertEqual(dn.run_day(args), 0)
            compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
