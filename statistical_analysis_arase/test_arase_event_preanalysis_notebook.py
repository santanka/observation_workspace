import json
import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


NOTEBOOK = Path(__file__).with_name("Arase_event_preanalysis.ipynb")


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cell_containing(notebook, marker):
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and marker in source:
            return source
    raise AssertionError(f"code cell not found: {marker}")


class EventPreanalysisNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook()

    def test_all_code_cells_compile(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")

    def test_mgf_missing_hours_are_not_treated_as_download_errors(self):
        source = code_cell_containing(self.notebook, "def load_mgf64_for_ranges")
        self.assertIn('No links matching pattern', source)
        self.assertIn('absence was not confirmed by remote index', source)
        self.assertIn('"missing_hours": list(missing_hours)', source)
        self.assertIn('"status"] = "missing_remote"', source)
        self.assertIn('"partial_missing" if missing_hours else "success"', source)
        self.assertIn("loaded_indices = sorted(loaded_blocks)", source)

    def test_mgf_remote_missing_hours_are_merged_into_hour_ranges(self):
        source = code_cell_containing(self.notebook, "def load_mgf64_for_ranges")
        start = source.index("def _mgf_missing_ranges")
        end = source.index("def load_mgf64_for_ranges")
        namespace = {"pd": pd}
        exec(source[start:end], namespace)

        result = namespace["_mgf_missing_ranges"]([
            pd.Timestamp("2021-01-01 03:00:00"),
            pd.Timestamp("2021-01-01 04:00:00"),
            pd.Timestamp("2021-01-01 07:00:00"),
        ])

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["gap_start_time"], pd.Timestamp("2021-01-01 03:00:00"))
        self.assertEqual(result.iloc[0]["gap_end_time"], pd.Timestamp("2021-01-01 05:00:00"))
        self.assertEqual(result.iloc[1]["reason"], "mgf_remote_missing")

    def test_gap_detection_includes_coverage_edges(self):
        source = code_cell_containing(self.notebook, "def find_time_gap_ranges")
        namespace = {"np": np, "pd": pd, "xr": xr}
        exec(source, namespace)

        time_da = xr.DataArray(pd.to_datetime([
            "2021-01-01 00:05:00",
            "2021-01-01 00:05:01",
        ]), dims=("time",))
        coverage = pd.DataFrame({
            "start_time": [pd.Timestamp("2021-01-01 00:00:00")],
            "end_time": [pd.Timestamp("2021-01-01 00:10:00")],
        })
        result = namespace["find_time_gap_ranges"](
            time_da,
            threshold=pd.Timedelta(seconds=30),
            coverage_ranges_df=coverage,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["gap_start_time"], coverage.iloc[0]["start_time"])
        self.assertEqual(result.iloc[-1]["gap_end_time"], coverage.iloc[0]["end_time"])

    def test_mgf_gap_cell_uses_coverage_and_remote_missing_ranges(self):
        source = code_cell_containing(self.notebook, "mgf_sample_gap_ranges_df")
        self.assertIn("coverage_ranges_df=merged_ranges_df", source)
        self.assertIn("MGF_remote_missing_ranges_df", source)

    def test_remote_missing_daily_logs_are_merged_into_ranges(self):
        source = code_cell_containing(self.notebook, "def remote_missing_ranges_from_logs")
        start = source.index("def logs_confirm_remote_missing")
        end = source.index("def load_erg_with_retry")
        namespace = {"pd": pd, "re": re}
        exec(source[start:end], namespace)

        result = namespace["remote_missing_ranges_from_logs"](
            [
                "No links matching pattern fake_20220527_v??.cdf",
                "No links matching pattern fake_20220528_v??.cdf",
                "No links matching pattern fake_20220530_v??.cdf",
            ],
            "test_remote_missing",
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["gap_start_time"], pd.Timestamp("2022-05-27"))
        self.assertEqual(result.iloc[0]["gap_end_time"], pd.Timestamp("2022-05-29"))
        self.assertEqual(result.iloc[1]["reason"], "test_remote_missing")

    def test_prefetch_accepts_partial_remote_missing(self):
        source = code_cell_containing(self.notebook, "def prefetch_erg_downloads")
        helper_start = source.index("def logs_confirm_remote_missing")
        helper_end = source.index("def load_erg_with_retry")
        prefetch_start = source.index("def _normalize_download_files")
        prefetch_end = source.index("ERG_PREFETCH_SPECS")

        class NoUsableScienceData(RuntimeError):
            pass

        namespace = {
            "pd": pd,
            "re": re,
            "Path": Path,
            "datetime": datetime,
            "timedelta": timedelta,
            "time": time,
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "as_completed": as_completed,
            "NoUsableScienceData": NoUsableScienceData,
            "ERG_LOAD_RETRY_DELAYS_SEC": (0,),
            "ERG_PREFETCH_BLOCK_DAYS": 1,
            "ERG_PREFETCH_MAX_WORKERS": 2,
        }
        exec(source[helper_start:helper_end], namespace)
        exec(source[prefetch_start:prefetch_end], namespace)

        with tempfile.TemporaryDirectory() as temp_dir:
            local_file = Path(temp_dir) / "available.cdf"
            local_file.touch()

            def capture(_load_func, **kwargs):
                if kwargs["trange"][0].startswith("20220501"):
                    return [str(local_file)], []
                return [], ["No links matching pattern fake_20220502_v??.cdf"]

            namespace["call_with_thread_log_capture"] = capture
            files, missing = namespace["prefetch_erg_downloads"](
                [("test product", lambda **kwargs: None, {})],
                ["20220501/00:00:00", "20220503/00:00:00"],
                max_workers=2,
                block_days=1,
            )

        self.assertEqual(len(files), 2)
        self.assertEqual(len(missing["test product"]), 1)
        self.assertEqual(
            missing["test product"].iloc[0]["gap_start_time"],
            pd.Timestamp("2022-05-02"),
        )

    def test_confirmed_absent_hfa_month_becomes_no_data(self):
        source = code_cell_containing(self.notebook, "def prefetch_erg_downloads")
        helper_start = source.index("def logs_confirm_remote_missing")
        helper_end = source.index("def load_erg_with_retry")
        prefetch_start = source.index("def _normalize_download_files")
        prefetch_end = source.index("ERG_PREFETCH_SPECS")

        class NoUsableScienceData(RuntimeError):
            pass

        namespace = {
            "pd": pd,
            "re": re,
            "Path": Path,
            "datetime": datetime,
            "timedelta": timedelta,
            "time": time,
            "ThreadPoolExecutor": ThreadPoolExecutor,
            "as_completed": as_completed,
            "NoUsableScienceData": NoUsableScienceData,
            "ERG_LOAD_RETRY_DELAYS_SEC": (0,),
            "ERG_PREFETCH_BLOCK_DAYS": 7,
            "ERG_PREFETCH_MAX_WORKERS": 2,
        }
        exec(source[helper_start:helper_end], namespace)
        exec(source[prefetch_start:prefetch_end], namespace)

        def capture(_load_func, **kwargs):
            return [], [
                "Unauthorized: https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/"
                "satellite/erg/pwe/hfa/l3/2022/07/"
            ]

        namespace["call_with_thread_log_capture"] = capture
        with self.assertRaisesRegex(
            NoUsableScienceData, "PWE-HFA L3"
        ):
            namespace["prefetch_erg_downloads"](
                [("PWE-HFA L3", lambda **kwargs: None, {})],
                ["20220701/00:00:00", "20220801/00:00:00"],
                max_workers=2,
                block_days=7,
            )

    def test_confirmed_absent_month_rule_is_product_and_month_specific(self):
        source = code_cell_containing(
            self.notebook, "def is_confirmed_absent_product_month"
        )
        start = source.index("def logs_confirm_remote_missing")
        end = source.index("def load_erg_with_retry")
        namespace = {"pd": pd, "re": re, "datetime": datetime}
        exec(source[start:end], namespace)
        self.assertTrue(namespace["is_confirmed_absent_product_month"](
            "PWE-HFA L3",
            ["20220701/00:00:00", "20220707/23:59:59"],
        ))
        self.assertFalse(namespace["is_confirmed_absent_product_month"](
            "PWE-HFA L3",
            ["20220801/00:00:00", "20220807/23:59:59"],
        ))
        self.assertFalse(namespace["is_confirmed_absent_product_month"](
            "LEPi L2 omniflux",
            ["20220701/00:00:00", "20220707/23:59:59"],
        ))

    def test_parent_index_distinguishes_absent_month_from_unauthorized(self):
        source = code_cell_containing(
            self.notebook, "def confirmed_absent_months_from_logs"
        )
        start = source.index("def logs_confirm_remote_missing")
        end = source.index("def load_erg_with_retry")

        class FakeRequests:
            class RequestException(Exception):
                pass

            @staticmethod
            def get(_url, timeout):
                self.assertEqual(timeout, (10, 20))

                class Response:
                    status_code = 200
                    text = '<a href="06/">06/</a><a href="08/">08/</a>'

                return Response()

        namespace = {
            "pd": pd,
            "re": re,
            "datetime": datetime,
            "requests": FakeRequests,
        }
        exec(source[start:end], namespace)
        messages = [
            "Unauthorized: https://example.test/product/2022/07/",
            "Unauthorized: https://example.test/product/2022/08/",
        ]

        self.assertEqual(
            namespace["confirmed_absent_months_from_logs"](messages),
            {"202207"},
        )

    def test_remote_month_directory_404_is_confirmed_missing(self):
        source = code_cell_containing(
            self.notebook, "def confirmed_absent_months_from_logs"
        )
        start = source.index("def logs_confirm_remote_missing")
        end = source.index("def load_erg_with_retry")
        namespace = {
            "pd": pd,
            "re": re,
            "datetime": datetime,
            "requests": None,
        }
        exec(source[start:end], namespace)
        messages = [
            "Remote index not found: https://example.test/product/2022/11/"
        ]

        self.assertEqual(
            namespace["confirmed_absent_months_from_logs"](messages),
            {"202211"},
        )
        self.assertTrue(namespace["logs_confirm_remote_missing"](messages))

    def test_mgf_uses_confirmed_absent_month_directories(self):
        source = code_cell_containing(self.notebook, "def load_mgf64_for_ranges")
        self.assertIn("confirmed_absent_months_from_logs(records)", source)
        self.assertIn('hour.strftime("%Y%m") in absent_months', source)

    def test_each_partial_product_is_added_to_a_forbidden_range(self):
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        for description in (
            "PWE-EFD L2 spectrum",
            "PWE-EFD L2 64 Hz",
            "LEPe L2 omniflux",
            "LEPi L2 omniflux",
            "PWE-HFA L3",
        ):
            self.assertGreaterEqual(
                source.count(f'get_prefetch_missing_ranges("{description}")'),
                1,
            )
        self.assertIn("ATT_remote_missing_ranges_df", source)


if __name__ == "__main__":
    unittest.main()
