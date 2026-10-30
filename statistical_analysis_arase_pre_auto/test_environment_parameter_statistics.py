import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import plot_environment_parameter_statistics as env


def make_frame():
    rows = []
    for i in range(6):
        row = {"phase_mode": "all", "status": "ok", "range_id": i,
               "n_occupied_blocks": 10, "block_duration": "10s",
               "range_MLAT_deg_mean": (-1.) ** i * 10.}
        for parameter, spec in env.PARAMETERS.items():
            row[spec["prefix"] + "_median"] = -20. if parameter == "dst" else 1.
            row[spec["prefix"] + "_valid_fraction"] = 1.
        for component in "EBS":
            row[f"kappa_{component}"] = float(i)
            row[f"log10_X0_{component}"] = -float(i)
        rows.append(row)
    return pd.DataFrame(rows)


class EnvironmentStatisticsTests(unittest.TestCase):
    def test_absolute_vsys_x_folds_saved_medians_and_recomputes_bins_and_correlations(self):
        frame = make_frame()
        frame['range_perp_sys_speed_ms_median'] = [-25000., -15000., -5000., 0., 5000., 15000.]
        selected, _ = env.select_parameter(frame, 'abs_vsys_x', 'all')
        x = env.parameter_values(selected, 'abs_vsys_x')
        np.testing.assert_allclose(x, [25., 15., 5., 0., 5., 15.])
        edges = env.make_edges([selected], 'abs_vsys_x', 10.)
        np.testing.assert_allclose(edges, [0., 10., 20., 30.])
        table = env.aggregate(selected, 'abs_vsys_x', 'all', edges, 1)
        group = table.loc[table.family.eq('kappa') & table.component.eq('E')]
        self.assertEqual(group.n.tolist(), [3, 2, 1])
        self.assertTrue(group.x_source_transform.eq('abs').all())
        self.assertAlmostEqual(group.pearson_r.iloc[0], np.corrcoef(x, selected.kappa_E)[0, 1])
        expected = env.correlations.pair_correlations(x, selected.kappa_E)
        self.assertAlmostEqual(group.spearman_rho.iloc[0], expected['spearman_rho'])

    def test_vsys_x_preserves_signed_component_converts_units_and_uses_linear_bins(self):
        frame = make_frame()
        column = env.PARAMETERS['vsys_x']['prefix'] + '_median'
        frame[column] = [-25000., -15000., -5000., 0., 5000., 15000.]
        frame[env.PARAMETERS['vsys_x']['prefix'] + '_valid_fraction'] = 0.
        selected, _ = env.select_parameter(frame, 'vsys_x', 'all')
        self.assertEqual(len(selected), 6)
        np.testing.assert_allclose(env.parameter_values(selected, 'vsys_x'), [-25., -15., -5., 0., 5., 15.])
        edges = env.make_edges([selected], 'vsys_x', 10.)
        np.testing.assert_allclose(edges, [-30., -20., -10., 0., 10., 20.])
        table = env.aggregate(selected, 'vsys_x', 'all', edges, 1)
        group = table.loc[table.family.eq('kappa') & table.component.eq('E')]
        self.assertEqual(group.n.tolist(), [1, 1, 1, 2, 1])
        self.assertTrue(group.x_unit.eq('km/s').all())
        self.assertTrue(group.x_source_unit.eq('m/s').all())
        self.assertTrue(group.x_source_scale.eq(.001).all())
        self.assertAlmostEqual(group.pearson_r.iloc[0], np.corrcoef(env.parameter_values(selected, 'vsys_x'), selected.kappa_E)[0, 1])

    def test_vsys_x_unified_command_needs_no_preparation(self):
        import run_statistics as unified
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'inputs').mkdir()
            for name in ['arase_phase_master.csv', 'annual_ranges.json', 'provenance.json']:
                (root / 'inputs' / name).touch()
            args = unified.build_parser().parse_args(['--year', '2022', '--statistics-dir', str(root),
                '--product', 'dependence', '--parameter', 'vsys_x', '--vsys-x-bin-width-kms', '5'])
            unified.validate(args)
            steps = unified.build_steps(args)[-1]
            self.assertEqual([s['name'] for s in steps], ['plot_dependence'])
            command = steps[0]['command']
            self.assertEqual(command[command.index('--vsys-x-bin-width-kms') + 1], '5.0')

    def test_default_does_not_screen_low_unknown_or_missing_coverage(self):
        for parameter, spec in env.PARAMETERS.items():
            frame = make_frame()
            frame[spec["prefix"] + "_valid_fraction"] = [0., .1, .79, np.nan, .8, 1.]
            selected, diagnostic = env.select_parameter(frame, parameter, "all")
            self.assertEqual(len(selected), 6)
            self.assertEqual(diagnostic["invalid_or_low_valid_fraction"], 0)
            frame = frame.drop(columns=spec["prefix"] + "_valid_fraction")
            self.assertEqual(len(env.select_parameter(frame, parameter, "all")[0]), 6)
            frame.loc[0, spec["prefix"] + "_median"] = np.nan
            self.assertEqual(len(env.select_parameter(frame, parameter, "all")[0]), 5)
        self.assertIsNone(env.build_parser().parse_args(["--year", "2022"]).min_valid_fraction)

    def test_density_rejects_nonpositive_and_bad_coverage_but_dst_keeps_negative_zero(self):
        frame = make_frame()
        prefix = env.PARAMETERS["ne"]["prefix"]
        frame[prefix + "_median"] = [0., -1., np.inf, 1., 1., 1.]
        frame[prefix + "_valid_fraction"] = [1., 1., 1., .79, .8, np.nan]
        selected, diagnostic = env.select_parameter(frame, "ne", "all", .8)
        self.assertEqual(selected.range_id.tolist(), [4])
        self.assertEqual(diagnostic["nonpositive_median"], 2)
        frame[env.PARAMETERS["dst"]["prefix"] + "_median"] = [-99., -20., 0., 13., np.nan, np.inf]
        selected, _ = env.select_parameter(frame, "dst", "all", .8)
        self.assertEqual(selected.range_id.tolist(), [0, 1, 2, 3])

    def test_edges_include_extremes_and_use_requested_linear_or_log_spacing(self):
        for parameter, values in (("dst", [-100., 0., 20.]), ("ne", [.01, 1., 100.])):
            frame = make_frame().iloc[:3].copy()
            frame[env.PARAMETERS[parameter]["prefix"] + "_median"] = values
            width = 20. if parameter == "dst" else .5
            edges = env.make_edges([frame], parameter, width)
            self.assertLessEqual(edges[0], min(values))
            self.assertGreater(edges[-1], max(values))
            np.testing.assert_allclose(np.diff(edges if parameter == "dst" else np.log10(edges)), width)

    def test_summary_threshold_is_per_finite_parameter_and_does_not_remove_csv_stats(self):
        frame = make_frame()
        frame.loc[:1, "kappa_E"] = np.nan
        edges = env.make_edges([frame], "ne", .5)
        table = env.aggregate(frame, "ne", "all", edges, 5)
        e = table.loc[table.family.eq("kappa") & table.component.eq("E") & table.n.gt(0)].iloc[0]
        b = table.loc[table.family.eq("kappa") & table.component.eq("B") & table.n.gt(0)].iloc[0]
        self.assertEqual(e["n"], 4)
        self.assertFalse(e.shown)
        self.assertEqual(e["median"], 3.5)
        self.assertTrue(b.shown)
        np.testing.assert_allclose([b.q16, b.q84], np.percentile(np.arange(6), [16, 84]))

    def test_cli_all_parameters_preserves_shared_inputs_and_applies_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "inputs").mkdir()
            frame = make_frame()
            frame.loc[0, "n_occupied_blocks"] = 9
            master = root / "inputs/arase_phase_master.csv"
            frame.to_csv(master, index=False)
            original = master.read_bytes()
            (root / "inputs/provenance.json").write_text(json.dumps({
                "year": 2022, "analysis_version": env.batch.ANALYSIS_VERSION,
                "annual_master_sha256": env.batch.file_sha256(master),
            }))
            with mock.patch.dict(env.PLOT_CONFIG, {"dpi": 35}):
                self.assertEqual(env.main(["--year", "2022", "--statistics-dir", str(root),
                                           "--phase", "all"]), 0)
            for parameter in env.DEFAULT_PARAMETERS:
                output = root / "min_blocks_10" / f"{parameter}_statistics"
                self.assertEqual(len(list(output.glob("*.png"))), 1)
                self.assertEqual(len(list(output.glob("*.pdf"))), 1)
                table = pd.read_csv(next(output.glob("*.csv")))
                self.assertTrue(table.loc[table.n.gt(0), "n"].eq(5).all())
                self.assertTrue(table.loc[table.n.gt(0), "shown"].all())
                meta = json.loads((output / "provenance.json").read_text())
                self.assertEqual(meta["diagnostics"]["all"]["accepted_ranges"], 5)
                self.assertEqual(meta["min_bin_count"], 5)
                self.assertIsNone(meta["min_valid_fraction"])
                self.assertFalse(meta["valid_fraction_screening"])
            self.assertEqual(master.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
