import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_occurrence_probability as occurrence


def make_master():
    return pd.DataFrame([
        {
            "range_id": 1, "range_duration_sec": 600.0,
            "phase_mode": "all", "status": "ok",
            "range_L_90deg_mean": 4.2, "range_MLAT_deg_mean": 7.0,
            "range_MLT_circular_mean_hour": 21.2,
        },
        {
            "range_id": 2, "range_duration_sec": 300.0,
            "phase_mode": "all", "status": "insufficient_data",
            "range_L_90deg_mean": 4.7, "range_MLAT_deg_mean": 8.0,
            "range_MLT_circular_mean_hour": 21.8,
        },
        {
            "range_id": 3, "range_duration_sec": 120.0,
            "phase_mode": "all", "status": "ok",
            "range_L_90deg_mean": 5.1, "range_MLAT_deg_mean": 12.0,
            "range_MLT_circular_mean_hour": 22.2,
        },
    ])


def make_orbit_points():
    return pd.DataFrame({
        "time": pd.date_range("2022-09-01", periods=4, freq="6s"),
        "r_re": [3.5, 3.6, 4.0, 4.1],
        "mlat_deg": [7.0, 8.0, 12.0, 12.5],
        "mlt_hour": [21.2, 21.8, 22.2, 22.4],
        "l_90deg": [4.2, 4.7, 5.1, 5.2],
        "dwell_sec": [500.0, 700.0, 100.0, 200.0],
    })


class OccurrenceProbabilityTests(unittest.TestCase):
    def test_manifest_window_uses_full_input_period(self):
        payload = {"time_range_input": ["20220901/00:00:00", "20221001/00:00:00"]}
        start, end = occurrence.manifest_time_window(payload, [])
        self.assertEqual(start, pd.Timestamp("2022-09-01"))
        self.assertEqual(end, pd.Timestamp("2022-10-01"))

    def test_full_ok_range_duration_is_divided_by_dwelling_time(self):
        phase = occurrence.select_phase_ranges(make_master(), "all")
        table, diagnostics = occurrence.aggregate_occurrence(
            phase, make_orbit_points(), "l_mlat",
            l_edges=[4.0, 5.0, 6.0], coordinate_edges=[5.0, 10.0, 15.0],
        )
        first = table.loc[
            table["l_bin"].eq(0) & table["coordinate_bin"].eq(0)
        ].iloc[0]
        self.assertEqual(first["dwelling_duration_sec"], 1200.0)
        self.assertEqual(first["kaw_duration_sec"], 600.0)
        self.assertEqual(first["batch_range_duration_sec"], 900.0)
        self.assertEqual(first["n_dwelling_samples"], 2)
        self.assertEqual(first["n_batch_ranges"], 2)
        self.assertEqual(first["n_kaw_ranges"], 1)
        self.assertAlmostEqual(first["occurrence_probability"], 0.5)
        self.assertEqual(diagnostics["n_binned_ranges"], 3)

    def test_kaw_duration_cannot_exceed_dwelling_time(self):
        points = make_orbit_points()
        points.loc[:1, "dwell_sec"] = 10.0
        phase = occurrence.select_phase_ranges(make_master(), "all")
        with self.assertRaisesRegex(ValueError, "exceeds spacecraft dwelling"):
            occurrence.aggregate_occurrence(
                phase, points, "l_mlat",
                l_edges=[4.0, 5.0, 6.0], coordinate_edges=[5.0, 10.0, 15.0],
            )

    def test_bins_are_left_closed_and_right_open(self):
        frame = make_master().iloc[[0]].copy()
        frame["range_L_90deg_mean"] = 5.0
        frame["range_MLAT_deg_mean"] = 10.0
        phase = occurrence.select_phase_ranges(frame, "all")
        points = pd.DataFrame({
            "l_90deg": [5.0], "mlat_deg": [10.0], "mlt_hour": [18.0],
            "dwell_sec": [1000.0],
        })
        table, _ = occurrence.aggregate_occurrence(
            phase, points, "l_mlat", l_edges=[4.0, 5.0, 6.0],
            coordinate_edges=[5.0, 10.0, 15.0],
        )
        selected = table.loc[table["kaw_duration_sec"].gt(0)].iloc[0]
        self.assertEqual(selected["l_bin"], 1)
        self.assertEqual(selected["coordinate_bin"], 1)

    def test_l_mlat_uses_only_18_to_06_mlt(self):
        values = pd.Series([18.0, 23.9, 0.0, 5.9, 6.0, 12.0, 17.9, np.nan])
        np.testing.assert_array_equal(
            occurrence._in_wrapped_mlt_sector(values),
            [True, True, True, True, False, False, False, False],
        )

        frame = make_master().iloc[[0]].copy()
        frame["range_MLT_circular_mean_hour"] = 15.0
        phase = occurrence.select_phase_ranges(frame, "all")
        points = pd.DataFrame({
            "l_90deg": [4.2], "mlat_deg": [7.0], "mlt_hour": [15.0],
            "dwell_sec": [1000.0],
        })
        l_mlat, diagnostics = occurrence.aggregate_occurrence(
            phase, points, "l_mlat", l_edges=[4.0, 5.0],
            coordinate_edges=[5.0, 10.0],
        )
        self.assertEqual(l_mlat["dwelling_duration_sec"].sum(), 0.0)
        self.assertEqual(l_mlat["kaw_duration_sec"].sum(), 0.0)
        self.assertEqual(diagnostics["n_orbit_samples_outside_mlt_sector"], 1)
        self.assertEqual(diagnostics["n_ranges_outside_mlt_sector"], 1)

        l_mlt, _ = occurrence.aggregate_occurrence(
            phase, points, "l_mlt", l_edges=[4.0, 5.0],
            coordinate_edges=[12.0, 18.0],
        )
        self.assertEqual(l_mlt["dwelling_duration_sec"].sum(), 1000.0)
        self.assertEqual(l_mlt["kaw_duration_sec"].sum(), 600.0)

    def test_missing_orbit_download_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_file = Path(tmp) / "orb.cdf"
            local_file.write_text("test")
            responses = [[], [], [], [str(local_file)]]
            with mock.patch("ergpyspedas.erg.orb", side_effect=responses) as loader:
                with mock.patch.object(occurrence.time, "sleep"):
                    files = occurrence.ensure_orbit_files(
                        pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-02"),
                        Path(tmp), retries=3, retry_delays=(0,),
                    )
            self.assertEqual(files, [local_file])
            self.assertEqual(loader.call_count, 4)

    def test_dwelling_cache_roundtrip(self):
        points = make_orbit_points()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dwelling.npz"
            occurrence.write_dwelling_cache(
                path, points, "hash", pd.Timestamp("2022-09-01"),
                pd.Timestamp("2022-10-01"),
            )
            loaded = occurrence.read_dwelling_cache(
                path, "hash", pd.Timestamp("2022-09-01"),
                pd.Timestamp("2022-10-01"),
            )
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(loaded["dwell_sec"], points["dwell_sec"])

    def test_log_norm_masks_zero_values(self):
        arrays = {
            "dwelling_duration_hour": np.array([[1.0, 10.0]]),
            "kaw_duration_hour": np.array([[0.0, 0.1]]),
            "occurrence_percent": np.array([[0.0, 1.0]]),
        }
        plotted, covered = occurrence._masked_arrays(arrays)
        time_norm, probability_norm = occurrence._norms(arrays, covered)
        self.assertIsInstance(time_norm, occurrence.LogNorm)
        self.assertIsInstance(probability_norm, occurrence.LogNorm)
        self.assertTrue(plotted[1].mask[0, 0])
        self.assertTrue(plotted[2].mask[0, 0])
        self.assertFalse(np.ma.getmaskarray(plotted[0])[0, 0])


    def test_meridional_grid_matches_l_and_mlat_boundaries(self):
        fig, ax = plt.subplots()
        try:
            occurrence.add_meridional_grid(
                ax, np.array([3.0, 4.0, 5.0]),
                np.array([-20.0, -10.0, 0.0, 10.0, 20.0]),
            )
            self.assertEqual(len(ax.lines), 8)
            labels = {text.get_text() for text in ax.texts}
            self.assertEqual(labels, {"3", "5", "-10°", "+10°"})
        finally:
            plt.close(fig)


    def test_render_both_coordinate_systems_with_earth(self):
        phase = occurrence.select_phase_ranges(make_master(), "all")
        old = dict(occurrence.PLOT_CONFIG)
        try:
            occurrence.PLOT_CONFIG.update({
                "l_edges": np.array([4.0, 5.0, 6.0]),
                "mlat_edges_deg": np.array([5.0, 10.0, 15.0]),
                "mlt_edges_hour": np.array([20.0, 21.0, 22.0, 23.0]),
                "dpi": 60,
            })
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rectangular, _ = occurrence.aggregate_occurrence(
                    phase, make_orbit_points(), "l_mlat"
                )
                polar, _ = occurrence.aggregate_occurrence(
                    phase, make_orbit_points(), "l_mlt"
                )
                paths = occurrence.plot_l_mlat(
                    rectangular, "all", "test", root / "l_mlat"
                )
                paths += occurrence.plot_l_mlt(
                    polar, "all", "test", root / "l_mlt"
                )
                self.assertTrue(all(path.is_file() for path in paths))
        finally:
            occurrence.PLOT_CONFIG.clear()
            occurrence.PLOT_CONFIG.update(old)
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
