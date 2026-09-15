import unittest

import numpy as np
import pandas as pd

import plot_kappa_time_series as kappa_plot


class KappaTimeSeriesTests(unittest.TestCase):
    def test_detected_selection_midpoint_and_error(self):
        base = {
            "range_start": "2024-01-01T00:00:00",
            "range_end": "2024-01-01T00:10:00",
            "phase_mode": "all",
            "kappa_E": 1.0,
            "kappa_E_q16": 0.8,
            "kappa_E_q84": 1.3,
            "kappa_B": 2.0,
            "kappa_B_q16": 1.7,
            "kappa_B_q84": 2.4,
        }
        frame = pd.DataFrame([
            {**base, "status": "ok"},
            {**base, "status": "insufficient_data"},
        ])
        selected = kappa_plot.select_detected_results(frame)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.loc[0, "plot_time"], pd.Timestamp("2024-01-01T00:05:00"))
        np.testing.assert_allclose(
            kappa_plot.asymmetric_error(selected, "E"), [[0.2], [0.3]]
        )
        self.assertEqual(kappa_plot.phase_means(selected), (1.0, 2.0))

    def test_invalid_or_nonfinite_interval_is_excluded(self):
        frame = pd.DataFrame([{
            "range_start": "2024-01-01T00:00:00",
            "range_end": "2024-01-01T00:10:00",
            "phase_mode": "all", "status": "ok",
            "kappa_E": 1.0, "kappa_E_q16": 1.1, "kappa_E_q84": 1.3,
            "kappa_B": 2.0, "kappa_B_q16": 1.7, "kappa_B_q84": np.nan,
        }])
        self.assertTrue(kappa_plot.select_detected_results(frame).empty)


if __name__ == "__main__":
    unittest.main()
