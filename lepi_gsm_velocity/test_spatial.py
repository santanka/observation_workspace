"""Known-value mean/geometry checks and a synthetic end-to-end publication."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import spatial as s


class SpatialTests(unittest.TestCase):
    def test_mlt_mesh_and_hourly_bins(self):
        xx, yy = s.mlt_mesh(np.array([4.]), np.array([0., 6., 12., 18.]))
        np.testing.assert_allclose(xx[:, 0], [4., 0., -4., 0.], atol=1e-14)
        np.testing.assert_allclose(yy[:, 0], [0., 4., 0., -4.], atol=1e-14)
        acc = s.Accumulator(s.L_EDGES, s.MLT_EDGES)
        acc.add(np.full(5, 4.5), np.array([0., 6., 12., 18., 23.99]),
                np.tile([-1., 2., -3.], (5, 1)), np.ones(5, bool), '2022-01-01')
        np.testing.assert_array_equal(np.flatnonzero(acc.n[1]), [0, 6, 12, 18, 23])
        self.assertEqual(acc.n.sum(), 5)

    def test_dayside_axis_and_screen_white_black(self):
        import matplotlib.pyplot as plt
        for sector in ('night', 'day'):
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            s.configure_meridional(ax, s.Accumulator(), sector)
            self.assertEqual(ax.xaxis_inverted(), sector == 'day')
            fig.canvas.draw()
            pixels = np.asarray(fig.canvas.buffer_rgba())
            for screen_side, expected in ((-1., 255), (1., 0)):
                data_x = screen_side * (-.5 if ax.xaxis_inverted() else .5)
                px, py = ax.transData.transform((data_x, 0.))
                rgb = pixels[pixels.shape[0] - 1 - round(py), round(px), :3]
                np.testing.assert_array_equal(rgb, np.full(3, expected))
            plt.close(fig)

    def test_signed_and_mean_absolute_are_distinct(self):
        acc = s.Accumulator()
        v = np.array([[10., -4., 2.], [-10., 8., 6.], [999., 999., 999.]])
        acc.add(np.full(3, 4.5), np.full(3, 2.), v, np.array([1, 1, 0], bool), '2022-01-01')
        signed, absolute = acc.means()
        np.testing.assert_allclose(signed[1, 10], [0., 2., 4.])
        np.testing.assert_allclose(absolute[1, 10], [10., 6., 4.])
        self.assertEqual(acc.days[1, 10], 1)
        acc.add(np.array([4.5]), np.array([2.]), v[:1], np.array([True]), '2022-01-02')
        self.assertEqual(acc.days[1, 10], 2)

    def test_bin_boundaries_nonfinite_and_outside(self):
        acc = s.Accumulator()
        v = np.ones((5, 3)); v[4] = np.nan
        count = acc.add(np.array([3., 11., 11.01, 2.99, 4.]),
                        np.array([-50., 50., 0., 0., 0.]), v, np.ones(5, bool), '2022-01-01')
        self.assertEqual(count, 2)
        self.assertEqual(acc.n[0, 0], 1)
        self.assertEqual(acc.n[-1, -1], 1)

    def test_orbit_wrap_gap_and_invalid_slots(self):
        times = np.array([-1., 0., 5., 10., 15., 20., 30., 100.])
        ot = np.array([0., 10., 20., 100.])
        ov = np.array([[4., 0., 23.9], [6., 10., .1], [np.nan, 20., .2], [8., 0., 12.]])
        lv, lat, mlt, valid = s.orbit_at(times, ot, ov, max_gap=30.)
        self.assertEqual(valid.tolist(), [False, True, True, True, False, False, False, True])
        self.assertAlmostEqual(lv[2], 5.)
        self.assertAlmostEqual(lat[2], 5.)
        self.assertAlmostEqual(mlt[2], 0.)
        np.testing.assert_array_equal(s.sector_mask(np.array([0., 6., 12., 18.]), 'night'), [True, False, False, True])
        # A finite support gap is also rejected.
        _, _, _, valid = s.orbit_at(np.array([50.]), np.array([0., 100.]), ov[[0, 3]], 30.)
        self.assertFalse(valid[0])

    def test_synthetic_end_to_end_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'gsm'
            config = {'schema': 'lepi_gsm_observed_velocity_v1', 'units': 'km/s'}
            profile = {'profile_hash': 'synthetic', 'config': config}
            root.mkdir(); (root / 'profile.json').write_text(json.dumps(profile))
            start = s.parse('2022-01-01').timestamp()
            times = start + np.array([1., 2., 3., 4.])
            arrays = {'time_unix': times, 'moment_valid': np.array([True, True, False, True]),
                      'rotation_valid': np.ones(4, bool), 'manifest_json': json.dumps(profile)}
            for species in s.SPECIES:
                arrays[species + '__velocity_valid'] = np.ones(4, bool)
                arrays[species + '__velocity_gsm_observed'] = np.array([[10., 2., 3.], [-10., 4., 5.], [999., 0., 0.], [8., 6., 7.]])
            data = root / '2022/01/data/20220101/chunk.npz'
            data.parent.mkdir(parents=True); np.savez(data, **arrays)
            state_path = root / '2022/01/state/20220101/chunk.json'
            state_path.parent.mkdir(parents=True)
            state = dict(profile_hash='synthetic', status='partial', output_sha256=s.sha(data),
                         start_utc='2022-01-01T00:00:00+00:00', end_utc_exclusive='2022-01-01T00:05:00+00:00')
            state_path.write_text(json.dumps(state))
            orbit = Path(temp) / 'orbit'; orbit.write_text('synthetic orbit')
            support = (start + np.array([0., 2., 4., 5.]), np.array([[4.5, 2., 0.], [4.5, 2., 0.], [4.5, 2., 12.], [4.5, 2., 12.]]))
            output = Path(temp) / 'output'
            argv = ['--gsm-root', str(root), '--start', '2022-01-01', '--end', '2022-01-02',
                    '--output-dir', str(output), '--min-count', '1']
            with patch.object(s, 'orbit_path', return_value=orbit), patch.object(s, 'read_orbit', return_value=support):
                self.assertEqual(s.main(argv), 0)
                with self.assertRaises(FileExistsError):
                    s.main(argv)
                state['output_sha256'] = 'wrong'; state_path.write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    s.main(argv[:-3] + [str(Path(temp) / 'broken'), '--min-count', '1'])
            import csv
            with (output / 'FPDU_night.csv').open() as stream:
                occupied = [row for row in csv.DictReader(stream) if int(row['n_records'])]
            self.assertEqual(len(occupied), 1)
            self.assertEqual(int(occupied[0]['n_records']), 2)
            self.assertEqual(float(occupied[0]['mean_x_kms']), 0.)
            self.assertEqual(float(occupied[0]['mean_abs_x_kms']), 10.)
            with (output / 'FPDU_l_mlt.csv').open() as stream:
                mlt_rows = [row for row in csv.DictReader(stream) if int(row['n_records'])]
            self.assertEqual(sum(int(row['n_records']) for row in mlt_rows), 3)
            self.assertEqual({float(row['mlt_left_hour']) for row in mlt_rows}, {0., 12.})
            self.assertEqual(len(list(output.glob('*.png'))), 9)
            self.assertEqual(len(list(output.glob('*.pdf'))), 9)
            report = json.loads((output / 'provenance.json').read_text())
            self.assertEqual(report['diagnostics']['records_in_period'], 4)


if __name__ == '__main__':
    unittest.main()
