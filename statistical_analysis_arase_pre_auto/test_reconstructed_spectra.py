import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import plot_reconstructed_spectra as spectra
import run_statistics as unified
from test_environment_parameter_statistics import make_frame


class SpectraTests(unittest.TestCase):
    def test_common_amplitude_point_and_spread_recovered_from_same_pairs(self):
        points = pd.DataFrame({"component": "E", "log10_X0": np.arange(1., 7.),
                               "kappa": np.arange(1., 7.)})
        table, values = spectra.reconstruct(points, np.array([1., 10., 100.]))
        np.testing.assert_allclose(values["E"][:, 1], 0.)
        self.assertEqual(table.log10_X_std.iloc[1], 0.)
        self.assertEqual(table.log10_X_q84_minus_q16.iloc[1], 0.)
        np.testing.assert_allclose(table.log10_X_median, [3.5, 0., -3.5])
        # Quartile membership fixed by X0, so its ordering reverses after intersection.
        self.assertGreater(table.high_X0_quartile_log10_X_median.iloc[0],
                           table.low_X0_quartile_log10_X_median.iloc[0])
        self.assertLess(table.high_X0_quartile_log10_X_median.iloc[2],
                        table.low_X0_quartile_log10_X_median.iloc[2])

    def test_render_selection_grid_output_policy_and_saved_input_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            frame = make_frame()
            frame["range_uid"] = [f"range-{i}" for i in range(len(frame))]
            frame.loc[0, "n_occupied_blocks"] = 9
            master = inputs / "arase_phase_master.csv"
            frame.to_csv(master, index=False)
            original = master.read_bytes()
            (inputs / "annual_ranges.json").write_text("{}")
            (inputs / "provenance.json").write_text(json.dumps({"year": 2022,
                "analysis_version": spectra.relation.batch.ANALYSIS_VERSION,
                "annual_master_sha256": spectra.relation.batch.file_sha256(master)}))
            args = unified.build_parser().parse_args(["--year", "2022", "--statistics-dir", str(root),
                "--product", "spectra", "--plot-only", "--phase", "all"])
            steps = unified.build_steps(args)[-1]
            self.assertEqual([s["name"] for s in steps], ["plot_spectra"])
            with mock.patch.dict(spectra.PLOT_CONFIG, {"dpi": 30}), \
                 mock.patch.object(spectra.styling, "detail_grid", wraps=spectra.styling.detail_grid) as styled:
                spectra.main(steps[0]["command"][2:])
                self.assertEqual(styled.call_count, 6)
            output = root / "min_blocks_10/reconstructed_spectra"
            self.assertEqual(len(list(output.iterdir())), 5)
            table = pd.read_csv(next(p for p in output.glob("*.csv") if '.points.' not in p.name))
            self.assertTrue(table.n.eq(5).all())
            self.assertEqual(len(table), 201 * 3)
            metadata = json.loads(next(output.glob("*.provenance.json")).read_text())
            self.assertFalse(metadata["fit_domain_coverage_applied"])
            stamps = {p: p.stat().st_mtime_ns for p in output.iterdir()}
            with mock.patch.object(spectra, "plot", side_effect=AssertionError("skip")):
                spectra.main(steps[0]["command"][2:])
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in output.iterdir()})
            self.assertEqual(master.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
