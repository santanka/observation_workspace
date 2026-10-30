import unittest

import numpy as np
import pandas as pd

from statistics_phases import classify_phases
import plot_kappa_x0_statistics as spatial
import plot_ae_parameter_statistics as ae


class StatisticsPhaseTests(unittest.TestCase):
    def test_both_hemispheres_keep_original_measurements(self):
        original = pd.DataFrame({
            "phase_mode": ["north_traveling", "south_traveling"] * 2,
            "range_MLAT_deg_mean": [10., 10., -10., -10.],
            "kappa_E": [1., 2., 3., 4.],
        })
        result = classify_phases(original)
        self.assertEqual(result.phase_mode.tolist(), [
            "toward_ionosphere", "toward_magnetic_equator",
            "toward_magnetic_equator", "toward_ionosphere",
        ])
        pd.testing.assert_series_equal(result.kappa_E, original.kappa_E)
        pd.testing.assert_series_equal(result.source_phase_mode, original.phase_mode, check_names=False)
        pd.testing.assert_frame_equal(classify_phases(result), result)
        self.assertNotIn("source_phase_mode", original)

    def test_equator_and_missing_latitude_remain_unclassified(self):
        result = classify_phases(pd.DataFrame({
            "phase_mode": ["north_traveling"] * 4 + ["standing", "all"],
            "range_MLAT_deg_mean": [0., np.nan, np.inf, -np.inf, np.nan, 0.],
        }))
        self.assertEqual(result.phase_mode.tolist(), ["unclassified"] * 4 + ["standing", "all"])
        self.assertEqual(result.direction_classification_reason.tolist(), [
            "zero_mlat", "missing_or_nonfinite_mlat", "missing_or_nonfinite_mlat",
            "missing_or_nonfinite_mlat", "unchanged", "unchanged",
        ])

    def test_selectors_use_reclassified_samples(self):
        rows = []
        for mlat in [10., -10.]:
            for phase, value in [("north_traveling", 1.), ("south_traveling", 4.)]:
                row = {"phase_mode": phase, "status": "ok", "range_L_90deg_mean": 5.,
                       "range_MLAT_deg_mean": mlat, "range_MLT_circular_mean_hour": 23.,
                       ae.AE_COLUMN: 100., ae.AE_VALID_FRACTION_COLUMN: 1.,
                       ae.AE_Q16_COLUMN: 80., ae.AE_Q84_COLUMN: 120.}
                for component in "EBS":
                    row[f"kappa_{component}"] = value
                    row[f"log10_X0_{component}"] = -value
                rows.append(row)
        frame = pd.DataFrame(rows)
        for selector in [spatial._select_phase, ae.select_phase]:
            toward = selector(frame, "toward_ionosphere")
            self.assertEqual(toward.kappa_E.tolist(), [1., 4.])
            equator = selector(frame, "toward_magnetic_equator")
            self.assertEqual(equator.kappa_E.tolist(), [4., 1.])


if __name__ == "__main__":
    unittest.main()
