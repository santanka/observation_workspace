import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_kappa_x0_statistics as statistics


def make_master():
    rows = []
    for range_id, status, l_value, mlat, mlt, offset in (
        (1, "ok", 4.2, 7.0, 21.0, 0.0),
        (2, "ok", 4.4, 8.0, 15.0, 1.0),
        (3, "insufficient_data", 4.3, 7.5, 22.0, 2.0),
    ):
        row = {
            "range_id": range_id,
            "phase_mode": "all",
            "status": status,
            "range_L_90deg_mean": l_value,
            "range_MLAT_deg_mean": mlat,
            "range_MLT_circular_mean_hour": mlt,
        }
        for index, component in enumerate("EBS"):
            row[f"kappa_{component}"] = 1.0 + index + offset
            row[f"log10_X0_{component}"] = -3.0 + index + offset
        rows.append(row)
    return pd.DataFrame(rows)


class ParameterStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(statistics.PLOT_CONFIG)
        statistics.PLOT_CONFIG.update({
            "l_edges": np.array([4.0, 5.0]),
            "mlat_edges_deg": np.array([5.0, 10.0]),
            "mlt_edges_hour": np.array([12.0, 18.0, 24.0]),
            "min_bin_count": 1,
            "dpi": 60,
        })

    def tearDown(self):
        statistics.PLOT_CONFIG.clear()
        statistics.PLOT_CONFIG.update(self.old)
        plt.close("all")

    def test_status_ok_and_nightside_selection(self):
        phase = statistics._select_phase(make_master(), "all")
        l_mlat, diagnostics = statistics.aggregate_statistics(
            phase, "all", "l_mlat", "kappa"
        )
        e_row = l_mlat.loc[l_mlat["component"].eq("E")].iloc[0]
        self.assertEqual(e_row["n"], 1)
        self.assertEqual(e_row["median"], 1.0)
        self.assertEqual(diagnostics["n_status_ok_ranges"], 2)
        self.assertEqual(diagnostics["n_ranges_outside_mlt_sector"], 1)

        l_mlt, diagnostics = statistics.aggregate_statistics(
            phase, "all", "l_mlt", "kappa"
        )
        self.assertEqual(l_mlt.loc[l_mlt["component"].eq("E"), "n"].sum(), 2)
        self.assertEqual(diagnostics["n_ranges_outside_mlt_sector"], 0)

    def test_log10_x0_uses_saved_log_values(self):
        phase = statistics._select_phase(make_master(), "all")
        table, _ = statistics.aggregate_statistics(
            phase, "all", "l_mlat", "log10_x0"
        )
        values = table.set_index("component")["median"]
        self.assertEqual(values["E"], -3.0)
        self.assertEqual(values["B"], -2.0)
        self.assertEqual(values["S"], -1.0)

    def test_minimum_count_is_configurable(self):
        phase = statistics._select_phase(make_master(), "all")
        table, _ = statistics.aggregate_statistics(
            phase, "all", "l_mlat", "kappa"
        )
        l_edges = statistics.PLOT_CONFIG["l_edges"]
        coordinate_edges = statistics.PLOT_CONFIG["mlat_edges_deg"]
        shown = statistics._masked_component(
            table, "E", l_edges, coordinate_edges
        )
        self.assertFalse(np.ma.getmaskarray(shown)[0, 0])
        statistics.PLOT_CONFIG["min_bin_count"] = 2
        masked = statistics._masked_component(
            table, "E", l_edges, coordinate_edges
        )
        self.assertTrue(np.ma.getmaskarray(masked)[0, 0])

    def test_overall_summary_uses_only_ranges_in_displayed_bins(self):
        phase = statistics._select_phase(make_master(), "all")
        table, _ = statistics.aggregate_statistics(
            phase, "all", "l_mlat", "kappa"
        )
        summary = statistics.displayed_range_summary(
            phase, table, "l_mlat", "kappa", "E"
        )
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["median"], 1.0)

        statistics.PLOT_CONFIG["min_bin_count"] = 2
        hidden = statistics.displayed_range_summary(
            phase, table, "l_mlat", "kappa", "E"
        )
        self.assertEqual(hidden["n"], 0)

    def test_panel_title_contains_label_and_distribution_summary(self):
        title = statistics._summary_title(
            "kappa", "E",
            {"n": 3, "median": 2.0, "q16": 1.0, "q84": 3.0},
            "d",
        )
        self.assertTrue(title.startswith("(d)"))
        self.assertIn("median = 2 [1, 3]", title)

    def test_color_limits_use_configured_percentiles_per_parameter(self):
        low, high = statistics._limits(np.arange(101.0), (5.0, 95.0))
        self.assertAlmostEqual(low, 5.0)
        self.assertAlmostEqual(high, 95.0)

        phase = statistics._select_phase(make_master(), "all")
        tables = []
        for family in statistics.FAMILIES:
            table, _ = statistics.aggregate_statistics(
                phase, "all", "l_mlat", family
            )
            tables.append(table)
        limits = statistics.determine_color_limits(tables)
        self.assertNotEqual(limits["kappa"]["E"], limits["kappa"]["B"])

    def test_render_both_coordinate_systems(self):
        phase = statistics._select_phase(make_master(), "all")
        tables = {}
        for coordinate in statistics.POSITION_COLUMNS:
            for family in statistics.FAMILIES:
                tables[(coordinate, family)], _ = statistics.aggregate_statistics(
                    phase, "all", coordinate, family
                )
        limits = statistics.determine_color_limits(list(tables.values()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for key, table in tables.items():
                coordinate, family = key
                paths.extend(statistics.plot_statistics(
                    table, "all", coordinate, family, limits[family],
                    "test", root / f"{coordinate}_{family}",
                ))
            for coordinate in statistics.POSITION_COLUMNS:
                coordinate_tables = {
                    family: tables[(coordinate, family)]
                    for family in statistics.FAMILIES
                }
                paths.extend(statistics.plot_combined_statistics(
                    phase, coordinate_tables, "all", coordinate, limits,
                    "test", root / f"{coordinate}_combined",
                ))
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
