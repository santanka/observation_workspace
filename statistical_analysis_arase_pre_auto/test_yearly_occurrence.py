import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import local_orbit_dwelling as orbit
import plot_occurrence_probability as occurrence
import plot_yearly_occurrence as yearly
from test_plot_occurrence_probability import make_master, make_orbit_points


class YearlyOccurrenceTests(unittest.TestCase):
    def test_fold_sums_numerator_and_denominator_before_division(self):
        frame = make_master().iloc[[0, 2]].copy()
        frame["range_L_90deg_mean"] = 4.2
        frame["range_MLAT_deg_mean"] = [7., -7.]
        points = make_orbit_points()
        points["l_90deg"] = 4.2
        points["mlat_deg"] = [7., 8., -7., -8.]
        table, _ = occurrence.aggregate_occurrence(
            frame, points, "l_abs_mlat", l_edges=[4., 5.], coordinate_edges=[5., 10.])
        row = table.iloc[0]
        self.assertEqual(row.n_kaw_ranges, 2)
        self.assertEqual(row.kaw_duration_sec, 720.)
        self.assertEqual(row.dwelling_duration_sec, 1500.)
        self.assertAlmostEqual(row.occurrence_probability, .48)

    def test_counts_include_insufficient_rows_without_treating_them_as_fit_success(self):
        frame = make_master()
        frame["detected_duration_sec"] = [1., .5, 2.]
        table = yearly.range_table(frame, "l_abs_mlat")
        self.assertEqual(table.n_batch_ranges.sum(), 3)
        self.assertEqual(table.n_kaw_ranges.sum(), 2)
        self.assertEqual(table.kaw_duration_sec.sum(), 720.)
        self.assertEqual(table.n_fit_north.sum(), 2)

    def test_relative_classification_precedes_folding(self):
        frame = make_master().iloc[[0, 0]].copy()
        frame["range_id"] = [1, 2]
        frame["phase_mode"] = ["north_traveling", "south_traveling"]
        frame["range_MLAT_deg_mean"] = [7., -7.]
        selected = occurrence.select_phase_ranges(frame, "toward_ionosphere")
        table = yearly.range_table(selected, "l_abs_mlat")
        self.assertEqual(table.n_kaw_ranges.sum(), 2)
        self.assertEqual(table.n_fit_north.sum(), 1)
        self.assertEqual(table.n_fit_south.sum(), 1)

    def test_zero_denominator_stays_nan_and_excess_stops(self):
        table = pd.DataFrame({"kaw_duration_sec": [0., 10.]})
        result = yearly.attach_dwelling(table, np.array([0., 20.]), np.array([0, 2]))
        self.assertTrue(np.isnan(result.occurrence_probability.iloc[0]))
        self.assertEqual(result.occurrence_probability.iloc[1], .5)
        with self.assertRaises(ValueError):
            yearly.attach_dwelling(table, np.array([0., 5.]), np.array([0, 1]))

    def test_counts_render_without_orbit_or_network(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(occurrence.PLOT_CONFIG, {"dpi": 40}):
            for coordinate in yearly.COORDINATES:
                table = yearly.range_table(make_master(), coordinate)
                paths = yearly.plot_counts(table, "all", coordinate, "test", Path(temp) / coordinate, 3)
                self.assertTrue(all(path.stat().st_size > 0 for path in paths))
        plt.close("all")

    def test_probability_main_writes_provenance_and_reuses_validated_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            frames = []
            for phase in ["all", "north_traveling", "south_traveling", "standing"]:
                frame = make_master()
                frame["phase_mode"] = phase
                frames.append(frame)
            master_path = root / "inputs/arase_phase_master.csv"
            pd.concat(frames).to_csv(master_path, index=False)
            (root / "inputs/provenance.json").write_text(json.dumps({
                "year": 2022, "analysis_version": yearly.batch.ANALYSIS_VERSION,
                "annual_master_sha256": yearly.batch.file_sha256(master_path),
                "sources": [{"month": "2022-01", "status": "processed"},
                            {"month": "2022-06", "status": "no_data"}],
            }))
            cdf_path = root / "dummy.cdf"
            cdf_path.write_bytes(b"synthetic fingerprint only")
            sums = {}
            for coordinate, spec in occurrence.POSITION_COLUMNS.items():
                shape = (len(occurrence.PLOT_CONFIG["l_edges"]) - 1,
                         len(occurrence.PLOT_CONFIG[spec["edges_key"]]) - 1)
                sums[coordinate] = [np.full(shape, 10000.), np.ones(shape, dtype=int)]
            with mock.patch.object(orbit, "orbit_inventory", return_value=({pd.Timestamp("2022-01-01"): cdf_path}, [])), \
                    mock.patch.object(yearly, "dwelling_tables", return_value=(sums, {"orb_weighted_duration_sec": 86400., "orbit_samples": 14400})) as loader, \
                    mock.patch.object(yearly, "plot_counts") as draw_counts, \
                    mock.patch.object(occurrence, "plot_l_mlat"), \
                    mock.patch.object(occurrence, "plot_l_mlt"), \
                    mock.patch.dict(occurrence.PLOT_CONFIG, dict(occurrence.PLOT_CONFIG)):
                argv = ["--year", "2022", "--statistics-dir", str(root)]
                self.assertEqual(yearly.main(argv + ["--counts-only"]), 0)
                self.assertEqual(draw_counts.call_count, 12)
                self.assertEqual(yearly.main(["--start-month", "202201", "--end-month", "202212",
                                             "--statistics-dir", str(root)]), 0)
                self.assertEqual(yearly.main(argv), 0)
                self.assertEqual(yearly.main(argv + ["--min-occupied-blocks", "6"]), 0)
                self.assertEqual(draw_counts.call_count, 12)
                self.assertEqual(loader.call_count, 1)
            block_root = root / "min_blocks_10"
            metadata = json.loads((block_root / "occurrence_probability/provenance.json").read_text())
            self.assertEqual(metadata["selected_months"], ["2022-01"])
            self.assertEqual(metadata["excluded_months"], ["2022-06"])
            self.assertEqual(len(list((block_root / "event_counts").glob("*.csv"))), 12)
            self.assertEqual(len(list((block_root / "occurrence_probability").glob("*.csv"))), 12)
            name = "occurrence_probability/arase_kaw_occurrence_l_abs_mlat_all_20220101_20221231.csv"
            six = pd.read_csv(root / "min_blocks_6" / name)
            ten = pd.read_csv(block_root / name)
            np.testing.assert_array_equal(six.dwelling_duration_sec, ten.dwelling_duration_sec)
            self.assertEqual(six.kaw_duration_sec.sum(), 720.)
            self.assertEqual(ten.kaw_duration_sec.sum(), 120.)
            self.assertEqual(metadata["selection"]["min_occupied_blocks"], 10)

    def test_folded_probability_plot_renders(self):
        frame = make_master()
        points = make_orbit_points()
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(occurrence.PLOT_CONFIG, {"dpi": 40}):
            table, _ = occurrence.aggregate_occurrence(frame, points, "l_abs_mlat")
            paths = occurrence.plot_l_mlat(table, "all", "test", Path(temp) / "probability")
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_inventory_uses_requested_months_and_latest_local_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "ergsc/satellite/erg/orb/def/2022"
            base.mkdir(parents=True)
            for day in ["20220101_v04", "20220101_v05", "20220601_v05"]:
                (base / f"erg_orb_l2_{day}.cdf").touch()
            found, missing = orbit.orbit_inventory(["2022-01"], root)
            self.assertEqual(len(found), 1)
            self.assertEqual(len(missing), 30)
            self.assertTrue(next(iter(found.values())).name.endswith("v05.cdf"))
            self.assertTrue(all(day.month == 1 for day in missing))

    def test_cdf_native_cadence_fills_and_daily_boundary(self):
        times = pd.to_datetime(["2022-01-01T00:00:00", "2022-01-01T00:00:06",
                                "2022-01-01T00:00:40", "2022-01-02T00:00:00"])
        position = np.array([[4., 7., 21.], [4., -1e31, 21.], [4., -7., 21.], [4., 7., 21.]])
        cdf = mock.MagicMock()
        cdf.__enter__.return_value = cdf
        cdf.varget.side_effect = lambda name: {"epoch": np.arange(4), "pos_rmlatmlt": position,
                                               "pos_Lm": np.tile([5., 6., 7.], (4, 1))}[name]
        with mock.patch.object(orbit.cdflib, "CDF", return_value=cdf), mock.patch.object(
                orbit.cdflib.cdfepoch, "to_datetime", return_value=times.to_numpy()):
            points = orbit.read_orbit_day(Path("unused.cdf"), pd.Timestamp("2022-01-01"))
        self.assertEqual(points.dwell_sec.tolist(), [6., 0., 0.])
        self.assertTrue(np.isnan(points.mlat_deg.iloc[1]))
        self.assertTrue(points.l_90deg.eq(5.).all())


if __name__ == "__main__":
    unittest.main()
