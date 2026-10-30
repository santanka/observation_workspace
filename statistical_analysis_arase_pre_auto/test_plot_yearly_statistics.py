import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

import batch_arase as batch
import plot_yearly_statistics as yearly
import plot_kappa_x0_statistics as spatial
import plot_ae_parameter_statistics as ae


class YearlyStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ranges = self.root / "ranges"
        self.states = self.root / "states"
        self.ranges.mkdir()
        self.master_paths = []
        for month in range(1, 13):
            start = pd.Timestamp(year=2022, month=month, day=1)
            end = start + pd.offsets.MonthBegin(1)
            path = self.ranges / f"Arase_valid_time_ranges_{start:%Y%m%d}_000000_to_{end:%Y%m%d}_000000.json"
            payload = {"time_range_input": [str(start), str(end)], "ranges": []}
            if month > 2:
                payload.update(analysis_status="no_data", no_data_reason="required input absent")
                path.write_text(json.dumps(payload))
                continue
            lo, hi = start + pd.Timedelta(hours=1), start + pd.Timedelta(hours=2)
            payload["ranges"] = [{"range_id": 1, "start_time": lo.isoformat(), "end_time": hi.isoformat()}]
            path.write_text(json.dumps(payload))
            aggregate = self.states / batch.dataset_key(path, batch.file_sha256(path)) / "aggregate"
            aggregate.mkdir(parents=True)
            pd.DataFrame([{"range_id": 1, "status": "complete", "analysis_version": batch.ANALYSIS_VERSION}]).to_csv(aggregate / "range_status.csv", index=False)
            rows = []
            for phase in yearly.SOURCE_PHASES:
                row = {
                    "range_id": 1, "range_start": str(lo), "range_end": str(hi),
                    "phase_mode": phase, "status": "ok", "analysis_version": batch.ANALYSIS_VERSION,
                    "n_occupied_blocks": 12, "block_duration": "10s",
                    "range_L_90deg_mean": 5.5, "range_MLAT_deg_mean": 2.0,
                    "range_MLT_circular_mean_hour": 23.0,
                    ae.AE_COLUMN: 100.0 + month, ae.AE_VALID_FRACTION_COLUMN: 1.0,
                    ae.AE_Q16_COLUMN: 80.0, ae.AE_Q84_COLUMN: 120.0,
                }
                for component in "EBS":
                    row[f"kappa_{component}"] = float(month)
                    row[f"log10_X0_{component}"] = -float(month)
                rows.append(row)
            master_path = aggregate / "arase_phase_master.csv"
            pd.DataFrame(rows).to_csv(master_path, index=False)
            self.master_paths.append(master_path)

    def prepare(self):
        return yearly.prepare_year(2022, self.ranges, self.states)

    def test_month_local_ids_remain_distinct_and_pool_raw_ranges(self):
        master, manifest, provenance = self.prepare()
        self.assertEqual(master["range_id"].nunique(), 2)
        self.assertEqual(master["source_range_id"].unique().tolist(), [1])
        self.assertEqual(master["range_uid"].nunique(), 2)
        self.assertEqual(len(manifest["ranges"]), 2)
        self.assertEqual(sum(s["status"] == "no_data" for s in provenance["sources"]), 10)
        table, _ = spatial.aggregate_statistics(spatial._select_phase(master, "all"), "all", "l_mlat", "kappa")
        selected = table.loc[table["n"].gt(0)]
        self.assertEqual(len(selected), 3)
        self.assertTrue(selected["n"].eq(2).all())
        self.assertTrue(selected["median"].eq(1.5).all())
        frame = ae.select_phase(master, "all")
        table = ae.aggregate_ae_statistics(frame, "all", ae.make_ae_edges([frame]))
        self.assertTrue(table.loc[table["n"].gt(0), "n"].eq(2).all())

    def test_multi_year_pooling_keeps_globally_unique_ids(self):
        original = next(self.ranges.glob("*20220101_000000_to_20220201*"))
        payload = json.loads(original.read_text())
        payload["time_range_input"] = ["20230101/00:00:00", "20230201/00:00:00"]
        for item in payload["ranges"]:
            for key in ["start_time", "end_time"]:
                item[key] = (pd.Timestamp(item[key]) + pd.DateOffset(years=1)).isoformat()
        path = self.ranges / "Arase_valid_time_ranges_20230101_000000_to_20230201_000000.json"
        path.write_text(json.dumps(payload))
        aggregate = self.states / batch.dataset_key(path, batch.file_sha256(path)) / "aggregate"
        aggregate.mkdir(parents=True)
        frame = pd.read_csv(self.master_paths[0])
        for column in ["range_start", "range_end"]:
            frame[column] = (pd.to_datetime(frame[column]) + pd.DateOffset(years=1)).astype(str)
        frame.to_csv(aggregate / "arase_phase_master.csv", index=False)
        pd.read_csv(self.master_paths[0].with_name("range_status.csv")).to_csv(aggregate / "range_status.csv", index=False)
        master, manifest, provenance = yearly.prepare_period(
            pd.Timestamp("2022-01-01"), pd.Timestamp("2023-02-01"), self.ranges, self.states)
        self.assertEqual(master.range_id.nunique(), 3)
        self.assertEqual(master.source_range_id.unique().tolist(), [1])
        self.assertEqual(len(provenance["sources"]), 13)
        self.assertEqual(provenance["start_month"], "202201")
        self.assertEqual(provenance["end_month"], "202301")
        self.assertIsNone(provenance["year"])
        self.assertEqual(manifest["time_range_input"][-1], "20230201/00:00:00")

    def test_prepare_only_writes_reusable_plot_inputs(self):
        output = self.root / "output"
        yearly.main([
            "--year", "2022", "--ranges-dir", str(self.ranges),
            "--state-root", str(self.states), "--output-dir", str(output),
            "--prepare-only",
        ])
        _, ranges = batch.load_ranges(output / "inputs" / "annual_ranges.json")
        self.assertEqual(len(ranges), 2)
        master = pd.read_csv(output / "inputs" / "arase_phase_master.csv")
        self.assertEqual(len(master), 8)
        provenance = json.loads((output / "inputs" / "provenance.json").read_text())
        self.assertEqual(provenance["annual_master_sha256"], batch.file_sha256(output / "inputs" / "arase_phase_master.csv"))
        self.assertFalse((output / "ae_statistics").exists())
        block_provenance = json.loads((output / "min_blocks_10" / "provenance.json").read_text())
        self.assertEqual(block_provenance["selection"]["min_occupied_blocks"], 10)

    def test_threshold_change_reuses_unfiltered_inputs_and_forwards_selection(self):
        output = self.root / "output"
        common = ["--year", "2022", "--ranges-dir", str(self.ranges),
                  "--state-root", str(self.states), "--output-dir", str(output)]
        yearly.main(common + ["--prepare-only", "--min-occupied-blocks", "6"])
        before = {p: p.read_bytes() for p in (output / "inputs").iterdir()}
        with mock.patch.object(yearly, "prepare_period", side_effect=AssertionError("must reuse prepared inputs")), \
                mock.patch.object(spatial, "main") as spatial_plot, mock.patch.object(ae, "main") as ae_plot:
            yearly.main(common + ["--min-occupied-blocks", "10"])
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
        self.assertTrue((output / "min_blocks_6/provenance.json").is_file())
        self.assertTrue((output / "min_blocks_10/provenance.json").is_file())
        for plotter in [spatial_plot, ae_plot]:
            argv = plotter.call_args.args[0]
            self.assertEqual(argv[argv.index("--min-occupied-blocks") + 1], "10")
            self.assertIn("min_blocks_10", argv[argv.index("--output-dir") + 1])

    def test_incomplete_month_stops(self):
        path = self.master_paths[0].with_name("range_status.csv")
        frame = pd.read_csv(path)
        frame["status"] = "failed"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Unfinished"):
            self.prepare()

    def test_duplicate_phase_stops(self):
        path = self.master_paths[0]
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "four phases"):
            self.prepare()

    def test_excluded_without_version_is_recorded_and_not_sampled(self):
        path = self.master_paths[0].with_name("range_status.csv")
        frame = pd.read_csv(path)
        frame["status"] = "excluded"
        frame["analysis_version"] = None
        frame.to_csv(path, index=False)
        pd.read_csv(self.master_paths[0]).iloc[:0].to_csv(self.master_paths[0], index=False)
        master, _, provenance = self.prepare()
        self.assertEqual(len(master), 4)
        self.assertEqual(provenance["sources"][0]["excluded_without_analysis_version"], 1)

    def test_old_version_stops(self):
        path = self.master_paths[0]
        frame = pd.read_csv(path)
        frame["analysis_version"] = "old"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            self.prepare()

    def test_missing_month_is_not_silently_skipped(self):
        next(self.ranges.glob("*20221201_000000_to_20230101*" )).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_master_time_mismatch_stops(self):
        path = self.master_paths[0]
        frame = pd.read_csv(path)
        frame["range_start"] = "2022-01-02T00:00:00"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "time mismatch"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
