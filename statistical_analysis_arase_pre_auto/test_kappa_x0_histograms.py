import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import plot_kappa_x0_histograms as histograms
import run_statistics as unified
from test_environment_parameter_statistics import make_frame


def frame():
    result = make_frame()
    result['range_uid'] = [f'range-{i}' for i in range(len(result))]
    return result


class HistogramTests(unittest.TestCase):
    def test_independent_finite_values_and_existing_selection_are_used(self):
        master = frame()
        master.loc[0, 'log10_X0_E'] = np.nan
        master.loc[1, 'kappa_E'] = np.inf
        master.loc[2, 'n_occupied_blocks'] = 9
        master.loc[3, 'status'] = 'failed'
        selected = histograms.selection.select_events(histograms.phases.classify_phases(master), 10)
        points = histograms.histogram_points(selected, 'all')
        x = points.loc[points.family.eq('log10_x0') & points.component.eq('E')]
        kappa = points.loc[points.family.eq('kappa') & points.component.eq('E')]
        self.assertEqual(set(x.range_uid), {'range-1', 'range-4', 'range-5'})
        self.assertEqual(set(kappa.range_uid), {'range-0', 'range-4', 'range-5'})
        self.assertEqual(len(points.loc[points.component.eq('B')]), 8)
        master['phase_mode'] = 'north_traveling'
        classified = histograms.phases.classify_phases(master)
        ionosphere = histograms.histogram_points(classified, 'toward_ionosphere')
        self.assertTrue(ionosphere.range_uid.isin(['range-0', 'range-2', 'range-4']).all())

    def test_counts_cover_extremes_keep_sparse_bins_and_do_not_normalize(self):
        points = histograms.histogram_points(histograms.phases.classify_phases(frame()), 'all')
        edges = {(family, component): histograms.make_edges(
            points.loc[points.family.eq(family) & points.component.eq(component), 'value'], .5)
            for family in histograms.parameters.FAMILIES for component in 'EBS'}
        table = histograms.aggregate(points, edges, 6)
        for _, group in table.groupby(['family', 'component']):
            self.assertEqual(group['count'].sum(), 6)
            self.assertTrue(group.total_n.eq(6).all())
            self.assertTrue(group.n_nonfinite.eq(0).all())
            self.assertEqual(group['count'].max(), 1)
            values = points.loc[points.family.eq(group.family.iloc[0]) & points.component.eq(group.component.iloc[0]), 'value']
            np.testing.assert_allclose(group['median'], np.median(values))
            np.testing.assert_allclose(group.variance, np.var(values, ddof=1))
            np.testing.assert_allclose(group['std'], np.std(values, ddof=1))
            self.assertTrue(group.variance_ddof.eq(1).all())
        np.testing.assert_allclose(histograms.make_edges([2., 2.], .5), [2., 2.5])
        with self.assertRaises(ValueError):
            histograms.make_edges([1., 2.], 0.)

    def test_summary_uses_unbinned_values_and_singleton_variance_is_undefined(self):
        source = frame()
        source['kappa_E'] = [0., 0., 0., 0., 0., 10.]
        points = histograms.histogram_points(histograms.phases.classify_phases(source), 'all')
        edges = {(family, component): histograms.make_edges(
            points.loc[points.family.eq(family) & points.component.eq(component), 'value'], 100.)
            for family in histograms.parameters.FAMILIES for component in 'EBS'}
        table = histograms.aggregate(points, edges, 6)
        row = table.loc[table.family.eq('kappa') & table.component.eq('E')].iloc[0]
        self.assertEqual(row['median'], 0.)
        self.assertAlmostEqual(row.variance, np.var([0., 0., 0., 0., 0., 10.], ddof=1))
        single = points.iloc[:1]
        table = histograms.aggregate(single, edges, 1)
        self.assertTrue(table.variance.isna().all())
        self.assertTrue(table['std'].isna().all())
        self.assertEqual(table.drop_duplicates(['family', 'component'])['median'].notna().sum(), 1)

    def test_unified_histogram_only_needs_no_environment_enrichment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'inputs').mkdir()
            for name in ['arase_phase_master.csv', 'annual_ranges.json', 'provenance.json']:
                (root / 'inputs' / name).touch()
            args = unified.build_parser().parse_args(['--year', '2022', '--statistics-dir', str(root),
                '--product', 'histogram', '--hist-kappa-bin-width', '.1', '--phase', 'standing'])
            unified.validate(args)
            steps = unified.build_steps(args)[-1]
            self.assertEqual([s['name'] for s in steps], ['plot_histogram'])
            command = steps[0]['command']
            self.assertEqual(command[command.index('--kappa-bin-width') + 1], '0.1')
            self.assertIn('standing', command)
            self.assertNotIn('--min-bin-count', command)

    def test_render_shared_axes_skip_overwrite_and_preserve_master(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'inputs').mkdir()
            first = frame()
            second = first.copy()
            second['phase_mode'] = 'standing'
            second['kappa_E'] += 1.
            master = root / 'inputs/arase_phase_master.csv'
            pd.concat([first, second], ignore_index=True).to_csv(master, index=False)
            before = master.read_bytes()
            (root / 'inputs/provenance.json').write_text(json.dumps({'year': 2022,
                'analysis_version': histograms.batch.ANALYSIS_VERSION,
                'annual_master_sha256': histograms.batch.file_sha256(master)}))
            args = ['--year', '2022', '--statistics-dir', str(root), '--phase', 'all', '--phase', 'standing']
            captured = []
            original = histograms.plot
            def record(table, phase, edges, limits, *remaining):
                captured.append((edges, limits))
                return original(table, phase, edges, limits, *remaining)
            with mock.patch.dict(histograms.PLOT_CONFIG, {'dpi': 30}), \
                 mock.patch.object(histograms, 'plot', side_effect=record), \
                 mock.patch.object(histograms.styling, 'detail_grid', wraps=histograms.styling.detail_grid) as styled:
                histograms.main([*args, '--overwrite'])
                self.assertEqual(styled.call_count, 12)
            self.assertEqual(captured[0][1], captured[1][1])
            for key in captured[0][0]:
                np.testing.assert_array_equal(captured[0][0][key], captured[1][0][key])
            output = root / 'min_blocks_10/kappa_x0_histograms'
            self.assertEqual(len(list(output.iterdir())), 9)
            stamps = {p: p.stat().st_mtime_ns for p in output.iterdir()}
            with mock.patch.object(histograms, 'plot', side_effect=AssertionError('skip')):
                histograms.main([*args, '--skip-existing'])
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in output.iterdir()})
            with mock.patch.dict(histograms.PLOT_CONFIG, {'dpi': 30}):
                histograms.main([*args, '--overwrite'])
            self.assertTrue(all(p.stat().st_mtime_ns > stamps[p] for p in output.iterdir()))
            self.assertEqual(master.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
