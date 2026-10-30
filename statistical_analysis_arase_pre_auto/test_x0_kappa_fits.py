import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import fit_x0_kappa_relationship as fitting
import run_statistics as unified


def points():
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 90)
    return pd.DataFrame({"range_uid": [str(i) for i in range(len(x))], "component": "E",
                         "month": np.repeat(["202201", "202202", "202203"], 30),
                         "log10_X0": x, "kappa": .4 + .7 * x + .3 * x ** 2})


class FitTests(unittest.TestCase):
    def test_recover_quadratic_and_hold_out_whole_months(self):
        frame = points()
        table, predictions, folds = fitting.compare_models(frame, bootstrap=20)
        quad = table.loc[table.model.eq("quadratic_ols")].iloc[0]
        self.assertLess(quad.cv_rmse, 1e-10)
        self.assertAlmostEqual(quad.c, .3)
        self.assertAlmostEqual(quad.b, .7 + .6 * quad.x_reference)
        self.assertEqual(quad.bootstrap_success, 20)
        self.assertEqual(quad.cv_n, len(frame))
        linear = table.loc[table.model.eq("linear_ols")].iloc[0]
        self.assertGreater(linear.cv_rmse, .1)
        shifted = frame.copy()
        shifted.loc[shifted.month.eq("202201"), "kappa"] += 10
        _, after, _ = fitting.compare_models(shifted, bootstrap=0)
        before = predictions.loc[predictions.model.eq("quadratic_ols") & predictions.month.eq("202201")]
        after = after.loc[after.model.eq("quadratic_ols") & after.month.eq("202201")]
        np.testing.assert_allclose(before.cv_predicted, after.cv_predicted)
        self.assertTrue(folds.n.eq(30).all())

    def test_robust_fit_limits_outlier_effect_and_pivot_transform_is_exact(self):
        x = np.linspace(-2, 2, 100)
        y = 1 + .5 * x
        y[-1] += 50
        _, ordinary = fitting.fit_model(x, y, 1, reference=0)
        _, robust = fitting.fit_model(x, y, 1, robust=True, reference=0)
        self.assertLess(abs(robust[1] - .5), abs(ordinary[1] - .5))
        frame = points()
        summary, transformed = fitting.pivot_diagnostics(frame, [3., 10.])
        self.assertEqual(len(summary), 3)
        selected = transformed.loc[transformed.q_pivot.eq(10.)]
        np.testing.assert_allclose(selected.log10_X_pivot, frame.log10_X0 - frame.kappa)

    def test_unified_fit_only_and_end_to_end_outputs_preserve_master(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            frame = points()
            master = pd.DataFrame({"range_uid": frame.range_uid, "phase_mode": "all", "status": "ok",
                "n_occupied_blocks": 10, "block_duration": "10s",
                "range_start": [f"{m[:4]}-{m[4:]}-01T00:00:00" for m in frame.month]})
            for component in "EBS":
                master[f"log10_X0_{component}"] = frame.log10_X0
                master[f"kappa_{component}"] = frame.kappa
            path = inputs / "arase_phase_master.csv"
            master.to_csv(path, index=False)
            original = path.read_bytes()
            (inputs / "annual_ranges.json").write_text("{}")
            (inputs / "provenance.json").write_text(json.dumps({"year": 2022,
                "analysis_version": fitting.relation.batch.ANALYSIS_VERSION,
                "annual_master_sha256": fitting.relation.batch.file_sha256(path)}))
            args = unified.build_parser().parse_args(["--year", "2022", "--statistics-dir", str(root),
                "--product", "x0_kappa_fit", "--plot-only", "--phase", "all", "--bootstrap-months", "5"])
            steps = unified.build_steps(args)[-1]
            self.assertEqual([s["name"] for s in steps], ["fit_x0_kappa"])
            with mock.patch.dict(fitting.PLOT_CONFIG, {"dpi": 30}):
                fitting.main(steps[0]["command"][2:])
            output = root / "min_blocks_10/x0_kappa_fits"
            self.assertEqual(len(list(output.iterdir())), 10)
            for p in output.iterdir():
                self.assertGreater(p.stat().st_size, 0)
            stamps = {p: p.stat().st_mtime_ns for p in output.iterdir()}
            with mock.patch.object(fitting, "compare_models", side_effect=AssertionError("skip")):
                fitting.main(steps[0]["command"][2:])
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in output.iterdir()})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
