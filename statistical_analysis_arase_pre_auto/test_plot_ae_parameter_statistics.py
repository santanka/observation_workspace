import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_ae_parameter_statistics as ae_statistics


def make_master():
    rows = []
    for range_id, status, ae, valid_fraction, mlt, offset in (
        (1, "ok", 80.0, 1.0, 21.0, 0.0),
        (2, "ok", 240.0, 1.0, 15.0, 1.0),
        (3, "ok", 260.0, 0.5, 22.0, 2.0),
        (4, "insufficient_data", 90.0, 1.0, 23.0, 3.0),
    ):
        row = {
            "range_id": range_id,
            "phase_mode": "all",
            "status": status,
            "range_geomag_AE_nT_median": ae,
            "range_geomag_AE_nT_q16": ae - 10.0,
            "range_geomag_AE_nT_q84": ae + 10.0,
            "range_geomag_AE_nT_valid_fraction": valid_fraction,
            "range_MLT_circular_mean_hour": mlt,
        }
        for index, component in enumerate("EBS"):
            row[f"kappa_{component}"] = 1.0 + index + offset
            row[f"log10_X0_{component}"] = -3.0 + index + offset
        rows.append(row)
    return pd.DataFrame(rows)


class AEParameterStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(ae_statistics.PLOT_CONFIG)
        ae_statistics.PLOT_CONFIG.update({
            "ae_bin_width_nT": 200.0,
            "ae_min_nT": 0.0,
            "ae_max_nT": None,
            "min_bin_count": 1,
            "ae_min_valid_fraction": 0.8,
            "mlt_sector_hour": None,
            "show_ae_range_spread": False,
            "annotate_bin_count": True,
            "dpi": 60,
        })

    def tearDown(self):
        ae_statistics.PLOT_CONFIG.clear()
        ae_statistics.PLOT_CONFIG.update(self.old)
        plt.close("all")

    def test_selection_uses_ok_status_and_ae_quality(self):
        selected = ae_statistics.select_phase(make_master(), "all")
        self.assertEqual(selected["range_id"].tolist(), [1, 2])

        ae_statistics.PLOT_CONFIG["mlt_sector_hour"] = (18.0, 6.0)
        nightside = ae_statistics.select_phase(make_master(), "all")
        self.assertEqual(nightside["range_id"].tolist(), [1])

    def test_dynamic_edges_include_largest_value(self):
        selected = ae_statistics.select_phase(make_master(), "all")
        edges = ae_statistics.make_ae_edges([selected])
        np.testing.assert_array_equal(edges, [0.0, 200.0, 400.0])

    def test_aggregate_uses_range_as_equal_sample(self):
        selected = ae_statistics.select_phase(make_master(), "all")
        edges = ae_statistics.make_ae_edges([selected])
        table = ae_statistics.aggregate_ae_statistics(selected, "all", edges)
        e_rows = table.loc[
            table["family"].eq("kappa") & table["component"].eq("E")
        ].sort_values("ae_bin")
        self.assertEqual(e_rows["n"].tolist(), [1, 1])
        self.assertEqual(e_rows["median"].tolist(), [1.0, 2.0])
        self.assertEqual(e_rows["overall_n"].iloc[0], 2)
        self.assertAlmostEqual(e_rows["spearman_rho"].iloc[0], 1.0)

    def test_render(self):
        selected = ae_statistics.select_phase(make_master(), "all")
        edges = ae_statistics.make_ae_edges([selected])
        table = ae_statistics.aggregate_ae_statistics(selected, "all", edges)
        limits = ae_statistics.determine_y_limits([selected])
        with tempfile.TemporaryDirectory() as tmp:
            paths = ae_statistics.plot_ae_statistics(
                selected, table, "all", edges, limits, "test",
                Path(tmp) / "ae_statistics",
            )
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
