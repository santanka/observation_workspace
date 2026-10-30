import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from cdflib import cdfwrite, cdfepoch
import numpy as np
import pandas as pd

import prepare_geomag_derivative_statistics as derivative
import plot_environment_parameter_statistics as plotting
import run_environment_statistics as runner
from statistics_period import resolve_period


def write_cdf(path, variable, time, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cdfwrite.CDF(path, delete=True)
    epochs = cdfepoch.compute_epoch([[t.year, t.month, t.day, t.hour, t.minute, t.second, 0] for t in time])
    writer.write_var({"Variable": "Epoch", "Data_Type": cdfwrite.CDF.CDF_EPOCH,
                      "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []}, var_data=epochs)
    variables = variable if isinstance(variable, dict) else {variable: values}
    for name, data in variables.items():
        writer.write_var({"Variable": name, "Data_Type": cdfwrite.CDF.CDF_INT4,
                          "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
                         var_attrs={"DEPEND_0": "Epoch", "FILLVAL": 99999,
                                    "VALIDMIN": {"DST": -800, "SYM_H": -800, "AL_INDEX": -7000}.get(name, 0),
                                    "VALIDMAX": 500 if name in ("DST", "SYM_H") else 7000}, var_data=data)
    writer.close()


def prepared_fixture(root):
    time = pd.date_range("2022-01-01T02:00", periods=6, freq="min")
    source = root / "original/kappa_phase_summary_toroidal_test.csv"
    source.parent.mkdir()
    source.write_text("original fit remains unchanged\n")
    pd.DataFrame({"time": time}).to_csv(source.with_name("kappa_timeseries_toroidal_test.csv"), index=False)
    row = {"range_uid": "dataset:1", "phase_mode": "all", "status": "ok", "n_occupied_blocks": 10,
           "block_duration": "10s", "range_MLAT_deg_mean": 10., "source_summary": str(source),
           "range_start": time[0].isoformat(), "range_end": time[-1].isoformat(),
           "range_geomag_AE_nT_n_total": 6, "range_geomag_Dst_nT_n_total": 6}
    row.update({"range_geomag_SYM_H_nT_n_total": 6, "range_geomag_AL_nT_n_total": 6,
                "range_geomag_SYM_H_nT_median": 22.5, "range_geomag_AL_nT_median": -245.,
                "range_geomag_SYM_H_nT_valid_fraction": 1., "range_geomag_AL_nT_valid_fraction": 1.})
    for component in "EBS":
        row[f"kappa_{component}"] = 1.
        row[f"log10_X0_{component}"] = -2.
    inputs = root / "inputs"
    inputs.mkdir()
    master = inputs / "arase_phase_master.csv"
    pd.DataFrame([row]).to_csv(master, index=False)
    (inputs / "provenance.json").write_text(json.dumps({
        "year": 2022, "analysis_version": derivative.batch.ANALYSIS_VERSION,
        "annual_master_sha256": derivative.batch.file_sha256(master)}))
    local = root / "local"
    write_cdf(local / "omni/hro_1min/2022/omni_hro_1min_20220101_v01.cdf",
              {"AE_INDEX": np.arange(361, dtype=int)*2, "SYM_H": np.arange(361, dtype=int)-100,
               "AL_INDEX": -np.arange(361, dtype=int)*2},
              pd.date_range("2022-01-01", periods=361, freq="min"), None)
    write_cdf(local / "omni/hourly/2022/omni2_h0_mrg1hr_20220101_v01.cdf", "DST",
              pd.date_range("2022-01-01", periods=7, freq="h"), np.arange(7, dtype=int)*10-100)
    return local, master, source


class DerivativeTests(unittest.TestCase):
    def test_native_slopes_have_correct_sign_units_and_centered_endpoints(self):
        for cadence, lag, slope, expected in (("1h", "1h", -3., -3.), ("1min", "5min", 2., 120.)):
            time = pd.date_range("2022-01-01", periods=30, freq=cadence)
            rate = derivative.differentiate(pd.Series(np.arange(30)*slope, index=time), cadence, lag)
            np.testing.assert_allclose(rate.dropna(), expected)
            steps = int(pd.Timedelta(lag)/pd.Timedelta(cadence))
            self.assertTrue(rate.iloc[:steps].isna().all())
            self.assertTrue(rate.iloc[-steps:].isna().all())
        # Centered stencil divides by 2h, not h.
        time = pd.date_range("2022-01-01", periods=3, freq="h")
        rate = derivative.differentiate(pd.Series([10., 30., 50.], index=time), "1h", "1h")
        self.assertEqual(rate.iloc[1], 20.)

    def test_backward_mode_and_missing_interior_native_slots(self):
        time = pd.date_range("2022-01-01", periods=25, freq="min")
        series = pd.Series(np.arange(25, dtype=float), index=time)
        backward = derivative.differentiate(series, "1min", "5min", "backward")
        self.assertTrue(backward.iloc[:5].isna().all())
        np.testing.assert_allclose(backward.iloc[5:], 60.)
        # Missing interior sample invalidates derivative even when both endpoints are finite.
        missing = series.drop(time[12])
        rate = derivative.differentiate(missing, "1min", "5min")
        self.assertTrue(rate.loc[time[7]:time[17]].isna().all())
        self.assertEqual(rate.loc[time[6]], 60.)
        self.assertEqual(rate.loc[time[18]], 60.)

    def test_nearest_alignment_does_not_skip_invalid_slots_or_extrapolate(self):
        time = pd.date_range("2022-01-01", periods=3, freq="h")
        rate = pd.Series([1., np.nan, 3.], index=time)
        target = pd.to_datetime(["2022-01-01T00:10", "2022-01-01T01:10", "2022-01-01T03:00"])
        sampled = derivative.sample_rates(rate, target, "1h")
        self.assertEqual(sampled[0], 1.)
        self.assertTrue(np.isnan(sampled[1:]).all())
        with self.assertRaises(ValueError):
            derivative.lags(.5, 5.)

    def test_cdf_fill_values_and_valid_bounds_are_masked(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "test.cdf"
            write_cdf(p, "AE_INDEX", pd.date_range("2022-01-01", periods=5, freq="min"),
                      [10, 99999, -1, 7001, 20])
            series = derivative.read_native([p], "AE_INDEX")
            np.testing.assert_allclose(series, [10., np.nan, np.nan, np.nan, 20.], equal_nan=True)

    def test_prepare_resume_and_plot_derivatives_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, master, source = prepared_fixture(root)
            before = (master.read_bytes(), source.read_bytes())
            arguments = ["--year", "2022", "--statistics-dir", str(root), "--spedas-dir", str(local)]
            self.assertEqual(derivative.main(arguments), 0)
            output = root / "environment_inputs/geomag_derivatives" / derivative.config_key()
            path = output / "range_summary.csv"
            table = pd.read_csv(path)
            for parameter, expected in (("d_dst_dt", 10.), ("d_ae_dt", 120.), ("d_sym_h_dt", 60.), ("d_al_dt", -120.)):
                prefix = derivative.SPECS[parameter]["prefix"]
                self.assertAlmostEqual(table[prefix + "_median"].iloc[0], expected)
                self.assertEqual(table[prefix + "_valid_fraction"].iloc[0], 1.)
            # Complete signed results are reused without loading native arrays again.
            with mock.patch.object(derivative, "read_native", side_effect=AssertionError("should reuse")):
                self.assertEqual(derivative.main(arguments), 0)
            with mock.patch.dict(plotting.PLOT_CONFIG, {"dpi": 30}):
                self.assertEqual(plotting.main(["--year", "2022", "--statistics-dir", str(root),
                    "--parameter", "d_dst_dt", "--parameter", "d_ae_dt",
                    "--parameter", "d_sym_h_dt", "--parameter", "d_al_dt",
                    "--parameter", "sym_h", "--parameter", "al", "--phase", "all"]), 0)
            for parameter in derivative.SPECS:
                plots = root / "min_blocks_10" / f"{parameter}_statistics"
                self.assertEqual(len(list(plots.glob("*.png"))), 1)
                metadata = json.loads((plots / "provenance.json").read_text())
                self.assertEqual(metadata["derivative_configuration"]["difference_method"], "centered")
            for parameter, expected in (("sym_h", 22.5), ("al", -245.)):
                table = pd.read_csv(next((root / "min_blocks_10" / f"{parameter}_statistics").glob("*.csv")))
                occupied = table.loc[table.n.gt(0)]
                self.assertTrue((occupied.lower.le(expected) & occupied.upper.gt(expected)).all())
                self.assertTrue(table.overall_n.eq(1).all())
            self.assertEqual((master.read_bytes(), source.read_bytes()), before)

    def test_plan_does_not_load_cdf_and_latest_cached_version_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, _, _ = prepared_fixture(root)
            old = local / "omni/hro_1min/2022/omni_hro_1min_20220101_v01.cdf"
            newer = old.with_name(old.name.replace("v01", "v02"))
            newer.write_bytes(old.read_bytes())
            with mock.patch.object(derivative, "read_native", side_effect=AssertionError("plan only")):
                self.assertEqual(derivative.main(["--year", "2022", "--statistics-dir", str(root),
                    "--spedas-dir", str(local), "--plan-only"]), 0)
            plan = json.loads((root / "environment_inputs/geomag_derivatives" / derivative.config_key() / "plan.json").read_text())
            self.assertEqual(plan["native_files"]["d_ae_dt"], [str(newer)])


class PipelineTests(unittest.TestCase):
    def test_defaults_centered_windows_and_prepare_before_plot(self):
        args = runner.build_parser().parse_args(["--year", "2022"])
        steps = runner.commands(args, Path("/tmp/prepared"), resolve_period(args))
        self.assertEqual([s[0] for s in steps], ["prepare_geomag_derivatives", "prepare_beta_e", "plot_dependencies"])
        self.assertIn("centered", steps[0][1])
        self.assertIn("--parameter", steps[2][1])
        self.assertIn("centered_dst_1h_ae_5min", " ".join(steps[2][1]))
        self.assertEqual(len(runner.DEFAULT_PARAMETERS), 8)
        self.assertIn("d_sym_h_dt", steps[2][1])
        self.assertIn("al", steps[2][1])
        self.assertIsNone(args.min_valid_fraction)
        self.assertNotIn("--min-valid-fraction", steps[2][1])
        args.plan_only = True
        self.assertEqual(len(runner.commands(args, Path("/tmp/prepared"), resolve_period(args))), 2)

    def test_new_lags_are_forwarded_and_separate_summary_paths(self):
        args = runner.build_parser().parse_args(["--year", "2022", "--parameter", "d_sym_h_dt",
                                                 "--symh-lag-minutes", "10", "--al-lag-minutes", "3"])
        steps = runner.commands(args, Path("/tmp/prepared"), resolve_period(args))
        self.assertEqual([s[0] for s in steps], ["prepare_geomag_derivatives", "plot_dependencies"])
        command = steps[0][1]
        self.assertEqual(command[command.index("--symh-lag-minutes") + 1], "10.0")
        self.assertIn("symh_10min_al_3min", " ".join(steps[1][1]))
        with self.assertRaises(ValueError):
            derivative.lags(1., 5., .5, 5.)

    def test_failed_prepare_stops_before_plot_and_records_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(runner.selection, "load_prepared_inputs", return_value=({}, {})), \
                 mock.patch.object(runner.subprocess, "run", side_effect=[None, RuntimeError("failed beta")]) as execute:
                with self.assertRaises(RuntimeError):
                    runner.main(["--year", "2022", "--statistics-dir", str(root)])
            self.assertEqual(execute.call_count, 2)
            record = json.loads((root / "environment_inputs/pipeline/latest_run.json").read_text())
            self.assertEqual([s["status"] for s in record["steps"]], ["complete", "failed_or_interrupted"])


if __name__ == "__main__":
    unittest.main()
