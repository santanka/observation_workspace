import json
import tempfile
import unittest
from pathlib import Path

import run_event_preanalysis_monthly as runner


class MonthlyRunnerTests(unittest.TestCase):
    def test_default_interval_contains_33_calendar_months(self):
        ranges = runner.build_month_ranges(runner.DEFAULT_START, runner.DEFAULT_END)
        self.assertEqual(len(ranges), 33)
        self.assertEqual(ranges[0].key, "202004")
        self.assertEqual(ranges[0].end_text, "20200501/00:00:00")
        self.assertEqual(ranges[-1].key, "202212")
        self.assertEqual(ranges[-1].end_text, "20230101/00:00:00")

    def test_leap_february_uses_calendar_boundary(self):
        ranges = runner.build_month_ranges(
            "20200201/00:00:00", "20200301/00:00:00"
        )
        self.assertEqual(ranges[0].end_text, "20200301/00:00:00")

    def test_non_month_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.build_month_ranges(
                "20200402/00:00:00", "20200501/00:00:00"
            )

    def test_notebook_time_range_is_replaced_once(self):
        month = runner.build_month_ranges(
            "20200401/00:00:00", "20200501/00:00:00"
        )[0]
        source, count = runner.transform_notebook_source(
            "before = 1\ntime_range = ['old', 'old']\nafter = 2\n", month
        )
        self.assertEqual(count, 1)
        self.assertIn(
            "time_range = ['20200401/00:00:00', '20200501/00:00:00']",
            source,
        )


    def test_valid_ranges_directory_can_be_redirected(self):
        month = runner.build_month_ranges(
            "20200401/00:00:00", "20200501/00:00:00"
        )[0]
        target = Path("/tmp/monthly-valid-ranges")
        source, count = runner.transform_notebook_source(
            f'save_dir = Path("{runner.DEFAULT_VALID_RANGES_DIR}")',
            month,
            target,
        )
        self.assertEqual(count, 0)
        self.assertEqual(source, f"save_dir = Path({str(target)!r})")
    def test_complete_status_requires_matching_hashes_and_result(self):
        month = runner.build_month_ranges(
            "20200401/00:00:00", "20200501/00:00:00"
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result.json"
            result.write_text(json.dumps({"ranges": []}))
            status = {
                "status": "complete",
                "notebook_sha256": "notebook",
                "runner_sha256": "runner",
                "result_path": str(result),
            }
            self.assertTrue(
                runner.compatible_complete(status, "notebook", "runner", result)
            )
            self.assertFalse(
                runner.compatible_complete(status, "changed", "runner", result)
            )
            self.assertEqual(month.key, "202004")

    def test_no_data_status_is_also_compatible_terminal_state(self):
        month = runner.build_month_ranges(
            "20210801/00:00:00", "20210901/00:00:00"
        )[0]
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result.json"
            result.write_text(json.dumps({"analysis_status": "no_data", "ranges": []}))
            status = {
                "status": "no_data",
                "notebook_sha256": "notebook",
                "runner_sha256": "runner",
                "result_path": str(result),
            }
            self.assertTrue(
                runner.compatible_complete(status, "notebook", "runner", result)
            )

    def test_no_data_payload_has_empty_ranges_and_full_unavailable_interval(self):
        month = runner.build_month_ranges(
            "20210801/00:00:00", "20210901/00:00:00"
        )[0]
        reason = "required product unavailable for full interval: LEPi L2 omniflux"
        payload = runner.no_data_result_payload(month, reason)
        self.assertEqual(payload["analysis_status"], "no_data")
        self.assertEqual(payload["ranges"], [])
        self.assertEqual(payload["n_valid_ranges"], 0)
        self.assertEqual(payload["unavailable_ranges"][0]["reason"], reason)
        self.assertEqual(
            payload["unavailable_ranges"][0]["start_time"],
            "2021-08-01T00:00:00",
        )

    def test_extract_no_data_reason(self):
        error = RuntimeError(
            "traceback\n\x1b[31mNoUsableScienceData\x1b[39m: "
            "required product unavailable: HFA"
        )
        self.assertEqual(
            runner.extract_no_data_reason(error),
            "required product unavailable: HFA",
        )


if __name__ == "__main__":
    unittest.main()
