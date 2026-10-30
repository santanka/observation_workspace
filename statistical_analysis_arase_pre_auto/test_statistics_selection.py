import argparse
import unittest

import pandas as pd

import statistics_selection as selection


class SelectionTests(unittest.TestCase):
    def test_boundary_status_and_raw_input_preserved(self):
        frame = pd.DataFrame({"phase_mode": ["all"] * 5,
                              "status": ["ok"] * 4 + ["insufficient_data"],
                              "n_occupied_blocks": [6, 9, 10, 12, 100],
                              "block_duration": ["10s"] * 5})
        before = frame.copy(deep=True)
        self.assertEqual(selection.select_events(frame, 10).index.tolist(), [2, 3])
        self.assertEqual(selection.select_events(frame, 6).index.tolist(), [0, 1, 2, 3])
        self.assertEqual(selection.selection_metadata(frame, 10)["phase_counts"]["all"],
                         {"before": 4, "after": 2})
        pd.testing.assert_frame_equal(frame, before)

    def test_thresholds_and_block_units_are_checked(self):
        parser = argparse.ArgumentParser()
        selection.add_selection_argument(parser)
        self.assertEqual(parser.parse_args([]).min_occupied_blocks, 10)
        self.assertEqual(selection.selection_key(6), "min_blocks_6")
        self.assertEqual(selection.selection_key(10), "min_blocks_10")
        with self.assertRaises(ValueError):
            selection.minimum_blocks(5)
        for count, duration in [(float("nan"), "10s"), (10.5, "10s"), (10, "20s")]:
            with self.subTest(count=count, duration=duration), self.assertRaises(ValueError):
                selection.accepted_events(pd.DataFrame({"status": ["ok"],
                                          "n_occupied_blocks": [count], "block_duration": [duration]}), 10)


if __name__ == "__main__":
    unittest.main()
