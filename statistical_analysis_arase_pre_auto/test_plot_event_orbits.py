import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_event_orbits as event_orbits


def native_orbit():
    time = pd.date_range("2020-01-01T00:00:00", periods=11, freq="1min")
    seconds = (time - time[0]).total_seconds().to_numpy()
    return pd.DataFrame({
        "time": time,
        "x_gsm_re": 2.0 + seconds / 600.0,
        "y_gsm_re": 3.0 + seconds / 300.0,
        "z_gsm_re": 4.0 + seconds / 200.0,
        "x_sm_re": 2.5 + seconds / 600.0,
        "y_sm_re": 3.5 + seconds / 300.0,
        "z_sm_re": 4.5 + seconds / 200.0,
    })


def event():
    return {
        "range_id": 7,
        "start_time": "2020-01-01T00:00:30",
        "end_time": "2020-01-01T00:08:30",
    }


class EventOrbitTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(event_orbits.PLOT_CONFIG)
        event_orbits.PLOT_CONFIG.update({"dpi": 60, "formats": ("png", "pdf")})

    def tearDown(self):
        event_orbits.PLOT_CONFIG.clear()
        event_orbits.PLOT_CONFIG.update(self.old)
        plt.close("all")

    def test_required_days_handles_midnight(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2020-01-01T23:59:00",
            "end_time": "2020-01-02T00:01:00",
        }]
        self.assertEqual(event_orbits.required_days(ranges), [
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
        ])

    def test_event_boundaries_are_interpolated_exactly(self):
        points = event_orbits.event_orbit_points(native_orbit(), event())
        self.assertEqual(points["time"].iloc[0], pd.Timestamp(event()["start_time"]))
        self.assertEqual(points["time"].iloc[-1], pd.Timestamp(event()["end_time"]))
        self.assertAlmostEqual(points["x_gsm_re"].iloc[0], 2.05)
        self.assertAlmostEqual(points["x_sm_re"].iloc[-1], 3.35)

    def test_two_minute_markers_exclude_boundaries(self):
        points = event_orbits.event_orbit_points(native_orbit(), event())
        markers = event_orbits.marker_points(points, "gsm")
        self.assertEqual(markers["time"].tolist(), list(pd.to_datetime([
            "2020-01-01T00:02:30", "2020-01-01T00:04:30",
            "2020-01-01T00:06:30",
        ])))

    def test_event_zoom_limits_preserve_axis_direction(self):
        xlim, ylim = event_orbits._event_zoom_limits(
            [2.0, 2.01], [3.0, 3.02], reverse_x=True, reverse_y=False
        )
        self.assertGreater(xlim[0], xlim[1])
        self.assertLess(ylim[0], ylim[1])
        self.assertAlmostEqual(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))

    def test_overview_limits_are_square_and_preserve_axis_direction(self):
        xlim, ylim = event_orbits._equal_span_limits((4.0, -2.0), (-1.0, 1.0))
        self.assertGreater(xlim[0], xlim[1])
        self.assertLess(ylim[0], ylim[1])
        self.assertAlmostEqual(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))

    def test_cache_roundtrip_and_separate_coordinate_figures(self):
        points = event_orbits.event_orbit_points(native_orbit(), event())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache.npz"
            event_orbits.write_cache(cache, points, "abc")
            loaded = event_orbits.read_cache(cache, "abc")
            self.assertIsNotNone(loaded)
            self.assertIsNone(event_orbits.read_cache(cache, "other"))
            written = []
            for coordinate in event_orbits.COORDINATES:
                written.extend(event_orbits.plot_event(
                    loaded, event(), coordinate, root / coordinate
                ))
            self.assertEqual(len(written), 4)
            self.assertTrue(all(path.is_file() for path in written))


if __name__ == "__main__":
    unittest.main()
