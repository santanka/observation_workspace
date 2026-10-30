import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt

import run_statistics as unified
import statistics_plotting as plotting
import plot_environment_parameter_statistics as environment
import plot_kappa_x0_latitude_profiles as latitude
from test_environment_parameter_statistics import make_frame


def prepared(root):
    inputs = root / "inputs"
    inputs.mkdir()
    frame = make_frame()
    frame["range_L_90deg_mean"] = 4.2
    frame["range_MLAT_deg_mean"] = 7.
    frame["range_MLT_circular_mean_hour"] = 21.
    master = inputs / "arase_phase_master.csv"
    frame.to_csv(master, index=False)
    (inputs / "annual_ranges.json").write_text("{}")
    (inputs / "provenance.json").write_text(json.dumps({
        "year": 2022, "analysis_version": environment.batch.ANALYSIS_VERSION,
        "annual_master_sha256": environment.batch.file_sha256(master)}))
    return master


class UnifiedStatisticsTests(unittest.TestCase):
    def test_common_ae_statistics_preserve_legacy_csv_columns_and_values(self):
        import pandas as pd
        frame = make_frame()
        frame[environment.PARAMETERS["ae"]["prefix"] + "_median"] = [10., 100., 200., 300., 400., 500.]
        edges = environment.make_edges([frame], "ae", 200.)
        common = environment.aggregate(frame, "ae", "all", edges, 5)
        legacy = environment.ae.aggregate_ae_statistics(frame, "all", edges)
        pd.testing.assert_frame_equal(common[list(legacy.columns)], legacy, check_dtype=False)

    def test_minor_grid_matches_requested_style_on_linear_and_log_axes(self):
        for scale in ("linear", "log"):
            fig, ax = plt.subplots()
            ax.set_xscale(scale)
            ax.set_xlim(.1, 100.)
            plotting.detail_grid(ax)
            fig.canvas.draw()
            for axis in (ax.xaxis, ax.yaxis):
                self.assertTrue(axis.get_minor_ticks())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    self.assertTrue(tick.gridline.get_visible())
                    self.assertEqual(tick.gridline.get_alpha(), .3)
            plt.close(fig)

    def test_artifact_policy_preserves_complete_sets_and_rejects_partial_sets(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "figure"
            self.assertTrue(plotting.should_write(base, False))
            base.with_suffix(".png").write_text("png")
            with self.assertRaises(FileExistsError):
                plotting.should_write(base, False)
            for suffix in (".pdf", ".csv"):
                base.with_suffix(suffix).write_text(suffix)
            self.assertFalse(plotting.should_write(base, False))
            self.assertTrue(plotting.should_write(base, True))

    def test_all_products_default_include_beta_e_ae_and_independent_enrichment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared(root)
            args = unified.build_parser().parse_args(["--year", "2022", "--statistics-dir", str(root), "--overwrite"])
            _, _, _, steps = unified.build_steps(args)
            names = [step["name"] for step in steps]
            self.assertEqual(names, ["prepare_derivatives", "prepare_beta_e", "plot_spatial", "plot_latitude",
                                     "plot_dependence", "plot_x0_kappa", "fit_x0_kappa", "plot_spectra",
                                     "plot_occurrence", "plot_event_counts"])
            for step in steps:
                if step["name"].startswith("prepare"):
                    self.assertNotIn("--overwrite", step["command"])
                else:
                    self.assertIn("--overwrite", step["command"])
            dependencies = next(s["command"] for s in steps if s["name"] == "plot_dependence")
            self.assertIn("ae", dependencies)
            self.assertIn("beta_e", dependencies)
            self.assertNotIn("--min-valid-fraction", dependencies)
            beta = next(s["command"] for s in steps if s["name"] == "prepare_beta_e")
            self.assertEqual(beta[beta.index("--workers") + 1], "2")
            profile = next(s["command"] for s in steps if s["name"] == "plot_latitude")
            self.assertEqual(profile[profile.index("--min-bin-count")+1], "1")

    def test_plot_only_dry_run_does_not_prepare_or_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared(root)
            before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            with mock.patch.object(unified.subprocess, "run", side_effect=AssertionError("dry run")):
                self.assertEqual(unified.main(["--year", "2022", "--statistics-dir", str(root),
                    "--product", "dependence", "--parameter", "beta_e", "--plot-only", "--dry-run"]), 0)
            self.assertEqual(sorted(str(p.relative_to(root)) for p in root.rglob("*")), before)

    def test_unified_overwrite_and_skip_with_real_rendering_preserve_master_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master = prepared(root)
            original = master.read_bytes()
            cache = root / "environment_inputs/beta_e/ranges/success.json"
            cache.parent.mkdir(parents=True)
            cache.write_text("saved beta_e cache")
            arguments = ["--year", "2022", "--statistics-dir", str(root), "--plot-only",
                         "--product", "dependence", "--product", "latitude", "--parameter", "ae",
                         "--phase", "all", "--l-bin", "4", "--latitude-coordinate", "l_abs_mlat"]
            modules = {"plot_environment_parameter_statistics.py": environment,
                       "plot_kappa_x0_latitude_profiles.py": latitude}
            def execute(command, **kwargs):
                return modules[Path(command[1]).name].main(command[2:])
            with mock.patch.object(unified.subprocess, "run", side_effect=execute), \
                 mock.patch.dict(environment.PLOT_CONFIG, {"dpi": 30}), mock.patch.dict(latitude.PLOT_CONFIG, {"dpi": 30}):
                unified.main([*arguments, "--overwrite"])
                artifacts = [p for p in (root / "min_blocks_10").rglob("*") if p.suffix in (".png", ".pdf", ".csv")]
                self.assertEqual(len(artifacts), 6)
                stamps = {p: p.stat().st_mtime_ns for p in artifacts}
                with mock.patch.object(environment, "plot", side_effect=AssertionError("skip")), \
                     mock.patch.object(latitude, "plot_profile", side_effect=AssertionError("skip")):
                    unified.main(arguments)
                self.assertEqual({p: p.stat().st_mtime_ns for p in artifacts}, stamps)
                with mock.patch.object(environment, "plot", wraps=environment.plot) as rendered:
                    unified.main([*arguments, "--overwrite"])
                    self.assertEqual(rendered.call_count, 1)
            self.assertEqual(master.read_bytes(), original)
            self.assertEqual(cache.read_text(), "saved beta_e cache")
            self.assertTrue(all(p.stat().st_mtime_ns > stamps[p] for p in artifacts))


if __name__ == "__main__":
    unittest.main()
