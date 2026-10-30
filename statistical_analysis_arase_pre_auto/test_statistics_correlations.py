from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np

import statistics_correlations as correlations
import plot_environment_parameter_statistics as env
from test_environment_parameter_statistics import make_frame


class CorrelationTests(unittest.TestCase):
    def test_figure_replacement_keeps_previous_on_render_failure_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'figure.png'
            path.write_bytes(b'old')
            figure = mock.Mock()
            def fail(output, **kwargs):
                output.write_bytes(b'partial')
                raise RuntimeError('render failure')
            figure.savefig.side_effect = fail
            with self.assertRaises(RuntimeError):
                env.statistics_plotting.save_figure(figure, path)
            self.assertEqual(path.read_bytes(), b'old')
            self.assertEqual(list(Path(temporary).iterdir()), [path])
            figure.savefig.side_effect = lambda output, **kwargs: output.write_bytes(b'new')
            env.statistics_plotting.save_figure(figure, path)
            self.assertEqual(path.read_bytes(), b'new')
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_pearson_uses_displayed_log_coordinate_while_rank_is_preserved(self):
        y = np.arange(6.)
        x = 10. ** y
        raw = correlations.pair_correlations(x, y)
        log = correlations.pair_correlations(x, y, log_x=True)
        self.assertAlmostEqual(raw['spearman_rho'], 1.)
        self.assertAlmostEqual(log['spearman_rho'], 1.)
        self.assertAlmostEqual(log['pearson_r'], 1.)
        self.assertLess(raw['pearson_r'], .9)
        for x, y in (([1.], [1.]), ([1., 1.], [1., 2.]), ([1., 2.], [0., 0.])):
            stats = correlations.pair_correlations(x, y)
            self.assertTrue(np.isnan(stats['pearson_r']))
            self.assertTrue(np.isnan(stats['spearman_rho']))
        stats = correlations.pair_correlations([-1., 0., 1., 10., np.inf], [1., 2., 0., 1., 9.], log_x=True)
        self.assertEqual(stats['correlation_n'], 2)
        self.assertAlmostEqual(stats['pearson_r'], 1.)

    def test_environment_correlation_uses_range_pairs_not_bin_medians(self):
        frame = make_frame()
        frame[env.PARAMETERS['ne']['prefix'] + '_median'] = 10. ** np.arange(6.)
        frame.loc[0, 'kappa_E'] = np.nan
        edges = env.make_edges([frame], 'ne', 10.)  # All six ranges in one bin.
        table = env.aggregate(frame, 'ne', 'all', edges, 5)
        self.assertTrue(table.pearson_x_transform.eq('log10').all())
        self.assertTrue(table.loc[table.family.eq('kappa'), 'pearson_r'].gt(.999).all())
        self.assertTrue(table.loc[table.family.eq('log10_x0'), 'pearson_r'].lt(-.999).all())
        self.assertEqual(table.loc[table.family.eq('kappa') & table.component.eq('E'), 'correlation_n'].iloc[0], 5)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(env.PLOT_CONFIG, {'dpi': 30}):
            with mock.patch.object(env.plt, 'close'):
                env.plot(frame, table, 'ne', 'all', edges, env.ae.determine_y_limits([frame]),
                         'test', Path(temporary) / 'figure')
                figure = plt.gcf()
                self.assertTrue(all(r'r_{\mathrm{P}}' in ax.get_title() for ax in figure.axes))
                self.assertTrue(all('log$_{10}$ x' in ax.get_title() for ax in figure.axes))
            plt.close(figure)

    def test_legacy_ae_title_and_csv_include_both_coefficients(self):
        frame = make_frame()
        frame[env.ae.AE_COLUMN] = np.arange(6.) ** 2
        table = env.ae.aggregate_ae_statistics(frame, 'all', np.array([0., 30.]))
        expected = np.corrcoef(frame[env.ae.AE_COLUMN], frame.kappa_E)[0, 1]
        row = table.loc[table.family.eq('kappa') & table.component.eq('E')].iloc[0]
        self.assertAlmostEqual(row.pearson_r, expected)
        self.assertAlmostEqual(row.spearman_rho, 1.)
        title = env.ae._panel_title(table, 'kappa', 'E', 'a')
        self.assertIn(r'r_{\mathrm{P}}', title)
        self.assertIn(r'\rho_{\mathrm{S}}', title)


if __name__ == '__main__':
    unittest.main()
