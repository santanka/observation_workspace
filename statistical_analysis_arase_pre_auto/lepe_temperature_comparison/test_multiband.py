"""Known-truth tests for PI quality, nonuniform time pairing and monthly routing."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import multiband as mb


def fit_frame(times, parallel=None, perpendicular=None):
    n = len(times)
    return pd.DataFrame(dict(ne=np.full(n, 1e6), T_para=np.full(n, 120.) if parallel is None else parallel,
                             T_perp=np.full(n, 60.) if perpendicular is None else perpendicular,
                             num_valid_data=np.full(n, 200), log_residual=np.full(n, .7),
                             eflux_para_mean=np.full(n, 2e4), eflux_perp_mean=np.full(n, 2e4)),
                        index=pd.DatetimeIndex(times, name='time_utc'))


class MultibandTests(unittest.TestCase):
    def test_component_masks_and_flux_defaults(self):
        f = fit_frame(pd.date_range('2022-09-01', periods=3, freq='1min'),
                      parallel=[120., np.nan, 240.])
        f.loc[f.index[2], 'eflux_para_mean'] = -100.
        low, high = mb.fit_quality(f, 1e4), mb.fit_quality(f)
        np.testing.assert_array_equal(low.effective_valid, [True, False, False])
        np.testing.assert_array_equal(low.perpendicular_valid, [True, True, True])
        np.testing.assert_array_equal(high.effective_valid, [True, False, True])
        self.assertTrue(high.parallel_nonpositive_flux_warning.iloc[2])
        self.assertEqual(low.parallel_reason.iloc[2], 4)
        self.assertEqual(high.effective_eV.iloc[0], 80.)

    def test_minimum_points_and_optional_temperature_ceiling(self):
        f = fit_frame(pd.date_range('2022-09-01', periods=2, freq='1min'), parallel=[120., 1e10])
        f.loc[f.index[0], 'num_valid_data'] = 3
        q = mb.fit_quality(f)
        self.assertEqual(q.parallel_reason.iloc[0], 2)
        self.assertEqual(q.parallel_eV.iloc[1], 1e10)
        q = mb.fit_quality(f, temperature_max=1e5)
        self.assertEqual(q.parallel_reason.iloc[1], 8)
        self.assertEqual(q.perpendicular_reason.iloc[1], 0)

    def test_nonuniform_raw_matching_uses_nanoseconds_and_rejects_large_offset(self):
        base = pd.Timestamp('2022-09-01')
        target = pd.DatetimeIndex([base + pd.Timedelta(seconds=s) for s in [0, 8, 24, 40]])
        # Explicit us-resolution PI index catches pandas us/ns integer conversion mistakes.
        source = pd.DatetimeIndex([base + pd.Timedelta(seconds=s, microseconds=200)
                                   for s in [0, 8, 24, 48]]).as_unit('us')
        q = mb.fit_quality(fit_frame(source))
        result = mb.match_raw(target, q)
        self.assertEqual(result.status.tolist(), ['matched'] * 3 + ['outside_tolerance'])
        np.testing.assert_allclose(result.nearest_offset_s.iloc[:3], .0002)
        self.assertTrue(np.isnan(result.fit_effective_eV.iloc[-1]))
        self.assertEqual(result.pi_source_index.iloc[-1], -1)

    def test_one_to_one_rejects_competing_matches_and_empty_input(self):
        base = pd.Timestamp('2022-09-01')
        target = pd.DatetimeIndex([base, base + pd.Timedelta(microseconds=500)])
        f = mb.fit_quality(fit_frame([base + pd.Timedelta(microseconds=200)]))
        result = mb.match_raw(target, f)
        self.assertEqual(result.status.tolist(), ['ambiguous', 'ambiguous'])
        self.assertTrue(result.fit_effective_eV.isna().all())
        empty = mb.match_raw(target, f.iloc[:0])
        self.assertEqual(empty.status.tolist(), ['no_raw_samples'] * 2)

    def test_component_native_recovers_independent_components(self):
        index = pd.date_range('2022-09-01', periods=2, freq='8s')
        n = pd.DataFrame(dict(magnetic_valid=[True, True], packet_info_raw=[0, 2],
                              density_cm3=[1., 1.]), index=index)
        d = {'raw_temperature_eV': np.array([[80., np.nan, 60.], [80., 120., 60.]])}
        result = mb.component_native(n, d, 2)
        self.assertTrue(result.effective_valid.iloc[0])
        self.assertFalse(result.parallel_valid.iloc[0])
        self.assertTrue(result.perpendicular_valid.iloc[0])
        self.assertTrue(result.filter(like='moment_').iloc[1].filter(like='_eV').notna().any())
        self.assertTrue(result[['moment_effective_eV', 'moment_parallel_eV', 'moment_perpendicular_eV']].iloc[1].isna().all())

    def test_minute_boundaries_and_component_coverage(self):
        lo, hi = pd.Timestamp('2022-09-01'), pd.Timestamp('2022-09-01T00:02')
        index = pd.to_datetime(['2022-09-01T00:00:59', '2022-09-01T00:01:00'])
        n = pd.DataFrame(index=index)
        n['valid'] = [False, True]
        for key in mb.core.COMPONENTS:
            n[f'moment_{key}_eV'] = [100., 200.]
            n[f'{key}_valid'] = [True, True]
        n.loc[index[0], 'moment_parallel_eV'] = np.nan
        n.loc[index[0], 'parallel_valid'] = False
        fit = mb.fit_quality(fit_frame(pd.date_range(lo, periods=2, freq='1min')))
        result = mb.minute_table(n, fit, lo, hi)
        np.testing.assert_allclose(result.moment_effective_eV, [100., 200.])
        np.testing.assert_array_equal(result.moment_parallel_n_valid, [0, 1])
        np.testing.assert_array_equal(result.moment_n_total, [1, 1])

    def test_loader_verifies_resolution_and_records_actual_energy_centers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fit.nc'
            frame = fit_frame(pd.date_range('2022-09-01', periods=2, freq='1min'))
            d = xr.Dataset({k: ('time', frame[k].values) for k in frame},
                           coords={'time': frame.index, 'energy': [963., 1178., 19076.]},
                           attrs={'fitmom_version': '2', 'resampling': '1min',
                                  'energy_min': 1000., 'energy_max': 20000.})
            d.to_netcdf(path)
            loaded, meta = mb.load_fit(path, 'high')
            self.assertEqual(len(loaded), 2)
            np.testing.assert_array_equal(meta['actual_energy_centers_eV'], [1178., 19076.])
            with self.assertRaises(ValueError):
                mb.load_fit(path, 'all')

    def test_display_cap_does_not_modify_input_and_marks_outliers(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values = np.array([80., 1e10])
        fig, ax = plt.subplots()
        count, maximum = mb.draw_series(ax, pd.date_range('2022-09-01', periods=2, freq='1min'),
                                         values, cap=1e5, linestyle='None')
        self.assertEqual((count, maximum), (1, 1e10))
        np.testing.assert_array_equal(values, [80., 1e10])
        self.assertEqual(ax.lines[-1].get_marker(), '^')
        plt.close(fig)

    def test_resume_rejects_corrupted_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in mb.OUTPUTS:
                (root / name).write_bytes(b'data')
            meta = dict(signature='signature', summary={'event': 'event'},
                        output_sha256={name: mb.core.sha256(root / name) for name in mb.OUTPUTS})
            (root / mb.PROVENANCE).write_text(json.dumps(meta))
            self.assertEqual(mb.reusable(root, 'signature'), meta['summary'])
            self.assertIsNone(mb.reusable(root, 'changed'))
            (root / mb.OUTPUTS[0]).write_bytes(b'corrupt')
            self.assertIsNone(mb.reusable(root, 'signature'))

    def test_month_forwards_range_options_and_continues_after_failed_day(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(month='2022-09', dataset=Path(temp) / 'dataset', output_root=Path(temp),
                                   raw_root=Path(temp), fit_1min_root=Path(temp), fit_raw_root=Path(temp),
                                   scratch=Path(temp), low_fit_flux_min=1e4, high_fit_flux_min=200.,
                                   raw_fit_flux_min=None, fit_temperature_max=None, plot_temperature_max=1e5,
                                   raw_match_tolerance_ms=1., force=False, dry_run=False, download_missing=True)
            processes = [SimpleNamespace(wait=lambda: 1), SimpleNamespace(wait=lambda: 0)]
            contexts = []
            from unittest.mock import MagicMock
            for process in processes:
                context = MagicMock()
                context.__enter__.return_value = process
                contexts.append(context)
            with patch.object(mb.core, 'discover_events', return_value=[None]), \
                 patch.object(mb.core, 'check_events'), patch.object(mb, 'input_files', return_value={}), \
                 patch.object(mb.subprocess, 'Popen', side_effect=contexts) as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mb.run_month(args, ['2022-09-01', '2022-09-02']), 1)
            for call in launch.call_args_list:
                command = call.args[0]
                self.assertIn('--download-missing', command)
                self.assertIn('--high-fit-flux-min', command)
                self.assertNotIn('--raw-fit-flux-min', command)
                self.assertEqual(command[command.index('--high-fit-flux-min') + 1], '200.0')
            summary = pd.read_csv(Path(temp) / 'dataset' / mb.core.OUTPUT_DIR / mb.VERSION / 'month_202209_summary.csv')
            self.assertEqual(summary.status.tolist(), ['failed', 'complete'])


if __name__ == '__main__':
    unittest.main()
