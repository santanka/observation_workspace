import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import plot_x0_kappa_relationship as relationship
import run_statistics as unified
from test_environment_parameter_statistics import make_frame


class RelationshipTests(unittest.TestCase):
    def frame(self):
        frame = make_frame()
        frame["range_uid"] = [f"range-{i}" for i in range(len(frame))]
        return frame

    def test_pairing_and_direction_use_same_range_and_component(self):
        frame = self.frame()
        frame["phase_mode"] = "north_traveling"
        frame.loc[0, "kappa_E"] = np.nan
        frame.loc[1, "log10_X0_E"] = np.inf
        frame.loc[2, "n_occupied_blocks"] = 9
        frame.loc[4, "status"] = "failed"
        classified = relationship.phases.classify_phases(frame)
        selected = relationship.selection.select_events(classified, 10)
        points = relationship.paired_points(selected, "toward_magnetic_equator")
        self.assertEqual(set(points.range_uid), {"range-1", "range-3", "range-5"})
        self.assertEqual(set(points.loc[points.component.eq("E"), "range_uid"]), {"range-3", "range-5"})
        self.assertTrue(points.source_phase_mode.eq("north_traveling").all())
        np.testing.assert_allclose(points.log10_X0, -points.kappa)

    def test_reference_amplitude_is_paired_and_rebinned_not_just_relabelled(self):
        frame = self.frame()
        for component in "EBS":
            frame[f"log10_X0_{component}"] = .1 + frame[f"kappa_{component}"] * np.log10(3.)
        points = relationship.paired_points(relationship.phases.classify_phases(frame), "all")
        transformed = relationship.reference_points(points, 3.)
        np.testing.assert_allclose(transformed.log10_X_reference, .1, atol=1e-14)
        np.testing.assert_allclose(transformed.log10_X0, points.log10_X0)
        edges = {c: relationship.make_edges(transformed.loc[transformed.component.eq(c), "log10_X_reference"], .5)
                 for c in "EBS"}
        table = relationship.aggregate(transformed, edges, 5, x_column="log10_X_reference")
        self.assertTrue(table.n.eq(6).all())
        self.assertTrue(table.shown.all())
        np.testing.assert_allclose(table["median"], 2.5)
        self.assertEqual(relationship.relationship_key(3.), "x0_kappa_relationship_kperp_rhoi_3")
        for invalid in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                relationship.relationship_key(invalid)

    def test_reference_three_render_and_unified_directory_preserve_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            frame = self.frame()
            master = inputs / "arase_phase_master.csv"
            frame.to_csv(master, index=False)
            original = master.read_bytes()
            (inputs / "annual_ranges.json").write_text("{}")
            (inputs / "provenance.json").write_text(json.dumps({"year": 2022,
                "analysis_version": relationship.batch.ANALYSIS_VERSION,
                "annual_master_sha256": relationship.batch.file_sha256(master)}))
            old = root / "min_blocks_10/x0_kappa_relationship/original.png"
            old.parent.mkdir(parents=True)
            old.write_bytes(b"original")
            args = unified.build_parser().parse_args(["--year", "2022", "--statistics-dir", str(root),
                "--product", "x0_kappa", "--phase", "all", "--plot-only", "--reference-kperp-rhoi", "3"])
            unified.validate(args)
            steps = unified.build_steps(args)[-1]
            command = steps[0]["command"]
            self.assertEqual(Path(command[command.index("--output-dir") + 1]).name,
                             "x0_kappa_relationship_kperp_rhoi_3")
            with mock.patch.dict(relationship.PLOT_CONFIG, {"dpi": 30}), \
                 mock.patch.object(relationship.styling, "detail_grid", wraps=relationship.styling.detail_grid) as styled:
                relationship.main(command[2:])
                self.assertEqual(styled.call_count, 3)
            output = root / "min_blocks_10/x0_kappa_relationship_kperp_rhoi_3"
            points = pd.read_csv(next(output.glob("*.points.csv")))
            np.testing.assert_allclose(points.log10_X_reference, points.log10_X0 - points.kappa * np.log10(3.))
            self.assertEqual(json.loads((output / "provenance.json").read_text())["reference_kperp_rhoi"], 3.)
            self.assertEqual(old.read_bytes(), b"original")
            self.assertEqual(master.read_bytes(), original)

    def test_bin_counts_spread_threshold_and_correlation_use_finite_pairs(self):
        frame = relationship.phases.classify_phases(self.frame())
        frame["log10_X0_E"] = [.1, .2, .3, .4, .45, 1.]
        frame.loc[0, "kappa_B"] = np.nan
        points = relationship.paired_points(frame, "all")
        edges = {c: relationship.make_edges(points.loc[points.component.eq(c), "log10_X0"], .5)
                 for c in "EBS"}
        table = relationship.aggregate(points, edges, 5)
        e = table.loc[table.component.eq("E")]
        self.assertEqual(e.n.tolist(), [5, 0, 1])
        self.assertEqual(e.shown.tolist(), [True, False, False])
        np.testing.assert_allclose(e.iloc[0][["q16", "median", "q84"]].to_numpy(float),
                                   np.percentile(np.arange(5), [16, 50, 84]))
        self.assertAlmostEqual(e.iloc[0].spearman_rho, 1.)
        self.assertAlmostEqual(e.iloc[0].pearson_r,
                               np.corrcoef(points.loc[points.component.eq('E'), 'log10_X0'],
                                           points.loc[points.component.eq('E'), 'kappa'])[0, 1])
        self.assertEqual(int(table.loc[table.component.eq("B"), "overall_n"].iloc[0]), 5)
        self.assertGreater(edges["E"][-1], points.log10_X0.max())

    def test_unified_selected_product_has_no_environment_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs").mkdir()
            for name in ("annual_ranges.json", "arase_phase_master.csv", "provenance.json"):
                (root / "inputs" / name).touch()
            args = unified.build_parser().parse_args(["--year", "2022", "--statistics-dir", str(root),
                "--product", "x0_kappa", "--x0-bin-width-dex", ".25", "--phase", "standing", "--overwrite"])
            unified.validate(args)
            steps = unified.build_steps(args)[-1]
            self.assertEqual([s["name"] for s in steps], ["plot_x0_kappa"])
            command = steps[0]["command"]
            self.assertEqual(command[command.index("--x0-bin-width-dex") + 1], "0.25")
            self.assertIn("standing", command)
            self.assertIn("--overwrite", command)

    def test_render_grid_skip_overwrite_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            frame = self.frame()
            # Coverage may be zero; it must not filter pairs.
            frame["range_ion_plasma_beta_HFA_valid_fraction"] = 0.
            master = inputs / "arase_phase_master.csv"
            frame.to_csv(master, index=False)
            original = master.read_bytes()
            (inputs / "provenance.json").write_text(json.dumps({"year": 2022,
                "analysis_version": relationship.batch.ANALYSIS_VERSION,
                "annual_master_sha256": relationship.batch.file_sha256(master)}))
            args = ["--year", "2022", "--statistics-dir", str(root), "--phase", "all"]
            with mock.patch.dict(relationship.PLOT_CONFIG, {"dpi": 30}), \
                 mock.patch.object(relationship.styling, "detail_grid", wraps=relationship.styling.detail_grid) as styled:
                relationship.main([*args, "--overwrite"])
                self.assertEqual(styled.call_count, 3)
            output = root / "min_blocks_10/x0_kappa_relationship"
            files = [p for p in output.iterdir() if p.name != "provenance.json"]
            self.assertEqual(len(files), 4)
            points = pd.read_csv(next(output.glob("*.points.csv")))
            self.assertEqual(len(points), 18)
            self.assertEqual(json.loads((output / "provenance.json").read_text())["grid_config"],
                             {"minor_ticks": True, "which": "both", "alpha": .3})
            stamps = {p: p.stat().st_mtime_ns for p in output.iterdir()}
            with mock.patch.object(relationship, "plot", side_effect=AssertionError("skip")):
                relationship.main([*args, "--skip-existing"])
            self.assertEqual({p: p.stat().st_mtime_ns for p in output.iterdir()}, stamps)
            with mock.patch.dict(relationship.PLOT_CONFIG, {"dpi": 30}):
                relationship.main([*args, "--overwrite"])
            self.assertTrue(all(p.stat().st_mtime_ns > stamps[p] for p in files))
            self.assertEqual(master.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
