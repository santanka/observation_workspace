"""Known-truth checks for tensor projection, temporal bins and quality selection."""
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import compare


class ComparisonTests(unittest.TestCase):
    def test_projection_is_rotation_invariant(self):
        # Gyrotropic known truth, rotated away from all coordinate axes.
        b = np.array([1., 2., 3.]) / np.sqrt(14.)
        tensor = 80 * np.eye(3) + (230 - 80) * np.outer(b, b)
        result = compare.project_temperature(tensor[None], (b * 100)[None])
        np.testing.assert_allclose(result, [[130., 230., 80.]])
        np.testing.assert_allclose(compare.project_temperature(tensor[None], (-b)[None]), result)

    def test_projection_general_tensor_and_zero_field(self):
        tensor = np.array([[[100., 20., 5.], [20., 200., 15.], [5., 15., 300.]]])
        np.testing.assert_allclose(compare.project_temperature(tensor, [[0., 1., 0.]]),
                                   [[200., 200., 200.]])
        self.assertTrue(np.isnan(compare.project_temperature(tensor, [[0., 0., 0.]])[0, 1:]).all())

    def test_magnetic_gap_and_extrapolation(self):
        out, valid, _, _, _ = compare.field_at_times(
            np.array([0., 8., 40.]), np.array([[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]]),
            np.array([-1., 4., 20., 40., 41.]))
        np.testing.assert_array_equal(valid, [False, True, False, True, False])
        np.testing.assert_allclose(out[1], [2., 3., 4.])

    def test_minutes_and_independent_component_masks(self):
        lo = pd.Timestamp('2022-09-01T00:00')
        hi = lo + pd.Timedelta(minutes=2)
        times = pd.to_datetime(['2022-09-01T00:00:59', '2022-09-01T00:01:00'])
        native = pd.DataFrame({f'moment_{k}_eV': [100., 200.] for k in compare.COMPONENTS}, index=times)
        native['valid'] = True
        fit = xr.Dataset({
            'T_para': ('time', [120., 240.]), 'T_perp': ('time', [60., 120.]),
            'eflux_para_mean': ('time', [2e4, 1e4]),
            'eflux_perp_mean': ('time', [2e4, 2e4]),
            'num_valid_data': ('time', [200, 200]), 'log_residual': ('time', [.5, .6]),
        }, coords={'time': pd.date_range(lo, periods=2, freq='1min')})
        result = compare.minute_comparison(native, fit, lo, hi, 1e4, None)
        np.testing.assert_allclose(result.moment_effective_eV, [100., 200.])
        self.assertEqual(result.fit_effective_eV.iloc[0], 80.)
        self.assertTrue(np.isnan(result.fit_effective_eV.iloc[1]))
        self.assertTrue(np.isnan(result.fit_parallel_eV.iloc[1]))
        self.assertEqual(result.fit_perpendicular_eV.iloc[1], 120.)
        np.testing.assert_array_equal(result.moment_n_total, [1, 1])

    def test_both_ranges_preserve_separate_validity_and_shared_fit(self):
        index = pd.date_range('2022-09-01', periods=2, freq='1min', name='time_utc')
        natives, tables = {}, {}
        for name, values in [('all', [200., np.nan]), ('le1000', [100., 150.])]:
            native = pd.DataFrame({f'moment_{k}_eV': values for k in compare.COMPONENTS}, index=index)
            native['valid'] = np.isfinite(values)
            table = native.drop(columns='valid').copy()
            table['moment_n_valid'] = native.valid.astype(int)
            table['fit_effective_eV'] = [120., 130.]
            natives[name], tables[name] = native, table
        native, table = compare.combine_profiles(natives, tables)
        self.assertTrue(np.isnan(table.moment_all_effective_eV.iloc[1]))
        self.assertEqual(table.moment_le1000_effective_eV.iloc[1], 150.)
        self.assertEqual(table.moment_all_n_valid.iloc[1], 0)
        self.assertEqual(table.moment_le1000_n_valid.iloc[1], 1)
        self.assertIn('all__valid', native)
        self.assertEqual(sum(k == 'fit_effective_eV' for k in table), 1)
        tables['le1000'].loc[index[0], 'fit_effective_eV'] = 999.
        with self.assertRaises(AssertionError):
            compare.combine_profiles(natives, tables)

    def test_month_selection_uses_existing_day_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['2022-08-31', '2022-09-01', '2022-09-30', '2022-10-01', 'orbit_coverage']:
                (root / name).mkdir()
            (root / '2022-09-02').write_text('not a directory')
            self.assertEqual(compare.selected_dates(root, month='2022-09'), ['2022-09-01', '2022-09-30'])
            self.assertEqual(compare.selected_dates(root, date='2022-09-30'), ['2022-09-30'])

    def test_resume_rejects_modified_or_missing_output_and_changed_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {compare.STEM + suffix: b'example output' for suffix in compare.OUTPUT_SUFFIXES}
            for name, data in files.items():
                (root / name).write_bytes(data)
            metadata = dict(signature='same', summary={'event': '0000-0010'},
                            output_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
            (root / f'{compare.STEM}_provenance.json').write_text(json.dumps(metadata))
            self.assertEqual(compare.reusable_result(root, 'same'), metadata['summary'])
            self.assertIsNone(compare.reusable_result(root, 'changed'))
            (root / (compare.STEM + '.png')).write_bytes(b'modified')
            self.assertIsNone(compare.reusable_result(root, 'same'))
            (root / (compare.STEM + '.png')).unlink()
            self.assertIsNone(compare.reusable_result(root, 'same'))

    def test_month_runs_separate_days_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(month='2022-09', dataset=Path(temp) / 'dataset', output_root=Path(temp),
                                   raw_root=Path(temp), fit_root=Path(temp), scratch=Path(temp),
                                   fit_flux_min=1e4, fit_temperature_max=None, force=False,
                                   dry_run=False, download_missing=True)
            events = [(None, None, None, None, None)]
            with patch.object(compare, 'discover_events', return_value=events), \
                 patch.object(compare, 'check_events'), patch.object(compare, 'input_files', return_value=[]), \
                 patch.object(compare.subprocess, 'run', side_effect=[SimpleNamespace(returncode=1),
                                                                     SimpleNamespace(returncode=0)]) as run:
                self.assertEqual(compare.run_month(args, ['2022-09-01', '2022-09-02']), 1)
            self.assertEqual(run.call_count, 2)
            for call, date in zip(run.call_args_list, ['2022-09-01', '2022-09-02']):
                command = call.args[0]
                self.assertEqual(command[command.index('--date') + 1], date)
                self.assertNotIn('--month', command)
                self.assertIn('--download-missing', command)
            summary = pd.read_csv(Path(temp) / 'dataset' / compare.OUTPUT_DIR / f'{compare.STEM}_month_202209_summary.csv')
            self.assertEqual(summary.status.tolist(), ['failed', 'complete'])

    def test_dry_run_never_acquires_missing_inputs(self):
        args = SimpleNamespace(date='2022-09-01', dataset=Path('/unused'),
                               dry_run=True, download_missing=True)
        with patch.object(compare, 'discover_events', return_value=[]), \
             patch.object(compare, 'check_events'), \
             patch.object(compare, 'input_files', side_effect=FileNotFoundError('missing MGF')), \
             patch.object(compare, 'acquire_missing_inputs') as acquire:
            with self.assertRaises(FileNotFoundError):
                compare.run_day(args)
            acquire.assert_not_called()


if __name__ == '__main__':
    unittest.main()
