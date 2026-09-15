import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import batch_arase as batch


class BatchAraseTests(unittest.TestCase):
    def test_manifest(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        self.assertEqual(len(ranges), 58)
        self.assertEqual(ranges[0]["range_id"], 1)
        self.assertEqual(ranges[-1]["range_id"], 58)

    def test_notebook_transform_syntax(self):
        notebook = json.loads(batch.DEFAULT_NOTEBOOK.read_text())
        transformed = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = batch.transform_notebook_source(
                "".join(cell["source"]),
                "2022-09-01T05:46:33.183191",
                "2022-09-01T06:21:52.179714",
                Path("/tmp/arase-auto-test"),
            )
            ast.parse(source)
            transformed.append(source)
        ast.parse(batch.CONTEXT_CELL)
        joined = "\n".join(transformed)
        self.assertIn("PLOT_WAVELET_MEDIAN_PSD = False", joined)
        self.assertIn("max_gap='5min'", joined)
        self.assertIn("bootstrap_samples_path =", joined)
        self.assertIn("/tmp/arase-auto-test", joined)
        self.assertNotIn(batch.NOTEBOOK_OUTPUT_ROOT, joined)
        self.assertEqual(joined.count("no_update=True"), 9)

    def test_hfa_interpolation_strict_five_minute_boundary(self):
        notebook = json.loads(batch.DEFAULT_NOTEBOOK.read_text())
        source = batch.transform_notebook_source(
            "".join(notebook["cells"][61]["source"]), "a", "b", Path("/tmp/test")
        )
        namespace = {"np": np, "pd": pd, "xr": xr}
        exec(source, namespace)
        interpolate = namespace["interp_unique_time"]
        t0 = pd.Timestamp("2022-01-01")

        for gap, midpoint_is_finite in (
            (pd.Timedelta(minutes=4, seconds=59), True),
            (pd.Timedelta(minutes=5), False),
        ):
            times = [t0, t0 + gap / 2, t0 + gap]
            source_da = xr.DataArray(
                [1.0, 3.0], dims="time", coords={"time": [t0, t0 + gap]}
            )
            target = xr.DataArray(times, dims="time", coords={"time": times})
            result = interpolate(source_da, target, max_gap="5min").values
            self.assertEqual(bool(np.isfinite(result[1])), midpoint_is_finite)
            self.assertEqual(result[0], 1.0)
            self.assertEqual(result[2], 3.0)

    def test_empty_aggregate(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {
                "state_dir": Path(tmp) / "state",
                "output_root": Path(tmp) / "output",
            })()
            batch.aggregate(args, ranges[:2])
            status = pd.read_csv(Path(tmp) / "state" / "aggregate" / "range_status.csv")
            self.assertEqual(status["status"].tolist(), ["pending", "pending"])

    def test_default_paths_are_namespaced_by_manifest_content(self):
        ranges_path = Path("/tmp/example ranges.json")
        args = type("Args", (), {
            "ranges": ranges_path,
            "state_dir": None,
            "output_root": None,
        })()
        key = batch.resolve_run_paths(args, "0123456789abcdef")
        self.assertEqual(key, "example_ranges_0123456789ab")
        self.assertEqual(args.state_dir, batch.DEFAULT_STATE_ROOT / key)
        self.assertEqual(args.output_root, batch.DEFAULT_OUTPUT_ROOT / key)

    def test_product_download_ranges(self):
        tranges = batch.download_tranges(
            "2022-09-01T23:30:00", "2022-09-02T00:20:00"
        )
        self.assertEqual(
            tranges["exact"],
            ["2022-09-01T23:30:00", "2022-09-02T00:20:00"],
        )
        self.assertEqual(
            tranges["omni_1min"],
            ["2022-09-01T20:30:00", "2022-09-02T03:20:00"],
        )
        self.assertEqual(
            tranges["omni_hourly"], ["2022-09-01", "2022-09-03"]
        )

    def test_aggregate_excludes_unaccepted_old_summary(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        item = ranges[0]
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            output_root = Path(tmp) / "output"
            summary = batch.expected_summary(item, output_root)
            summary.parent.mkdir(parents=True)
            pd.DataFrame({"phase": ["old"]}).to_csv(summary, index=False)
            args = type("Args", (), {
                "state_dir": state_dir,
                "output_root": output_root,
            })()
            batch.aggregate(args, [item])
            master = state_dir / "aggregate" / "arase_phase_master.csv"
            self.assertFalse(master.exists())


if __name__ == "__main__":
    unittest.main()
