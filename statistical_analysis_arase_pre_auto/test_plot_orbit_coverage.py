import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import plot_orbit_coverage as orbit
import matplotlib.pyplot as plt


class OrbitCoverageTests(unittest.TestCase):
    def test_default_output_mode_writes_both_figure_sets(self):
        args = orbit.build_parser().parse_args([])
        self.assertFalse(args.detected_only)
        self.assertFalse(args.all_ranges_only)

    def test_successful_range_ids_are_phase_specific(self):
        master = pd.DataFrame({
            "range_id": [1, 2, 1, 2],
            "phase_mode": ["all", "all", "standing", "standing"],
            "status": ["ok", "insufficient_data", "ok", "ok"],
            "kappa_E": [1.0, np.nan, 1.1, np.nan],
            "kappa_B": [2.0, np.nan, 2.1, 2.2],
        })
        self.assertEqual(orbit.successful_range_ids(master, "all"), {1})
        self.assertEqual(orbit.successful_range_ids(master, "standing"), {1})

    def test_day_night_earth_patches(self):
        fig, axes = plt.subplots(1, 2)
        orbit.add_earth(axes[0], "xy")
        orbit.add_earth(axes[1], "yz")
        self.assertEqual(len(axes[0].patches), 3)  # two hemispheres and outline
        self.assertEqual(len(axes[1].patches), 2)  # black disk and outline
        plt.close(fig)

    def test_plot_window_is_derived_for_another_year(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2024-02-03T12:00:00",
            "end_time": "2024-02-05T06:00:00",
        }]
        start, end = orbit.resolve_plot_window(ranges)
        self.assertEqual(start, pd.Timestamp("2024-02-03T00:00:00"))
        self.assertEqual(end, pd.Timestamp("2024-02-06T00:00:00"))

    def test_clipping_and_midnight_split(self):
        source = [{
            "range_id": 7,
            "start_time": "2022-09-01T23:30:00",
            "end_time": "2022-09-02T00:30:00",
        }]
        selected = orbit.clipped_ranges(
            source, pd.Timestamp("2022-09-01"), pd.Timestamp("2022-10-01")
        )
        by_day = orbit.ranges_by_day(selected)
        self.assertEqual(list(by_day), [
            pd.Timestamp("2022-09-01"), pd.Timestamp("2022-09-02")
        ])
        self.assertEqual(by_day[pd.Timestamp("2022-09-01")][0]["range_id"], 7)

    def test_cache_roundtrip_and_hash_rejection(self):
        points = pd.DataFrame({
            "time": pd.to_datetime(["2022-09-01T00:00:00", "2022-09-01T00:00:08"]),
            "x_gsm_re": [1.0, 2.0],
            "y_gsm_re": [3.0, 4.0],
            "z_gsm_re": [5.0, 6.0],
            "range_id": [1, 1],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orbit.npz"
            orbit.write_cache(path, points, "abc", "start", "end")
            loaded = orbit.read_cache(path, "abc", "start", "end")
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(
                loaded[["x_gsm_re", "y_gsm_re", "z_gsm_re"]],
                points[["x_gsm_re", "y_gsm_re", "z_gsm_re"]],
            )
            self.assertIsNone(orbit.read_cache(path, "different", "start", "end"))


if __name__ == "__main__":
    unittest.main()
