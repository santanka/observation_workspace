import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

import plot_environment_parameter_statistics as plotting
import prepare_electron_beta_statistics as beta


class ElectronBetaTests(unittest.TestCase):
    def test_worker_rejects_stale_parent_code_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            request = Path(temp) / "request.json"
            request.write_text(json.dumps({"notebook": str(beta.batch.DEFAULT_NOTEBOOK),
                                           "preparer_sha256": "stale"}))
            with self.assertRaisesRegex(RuntimeError, "source changed after planning"):
                beta.worker(request)

    def test_plasma_prefix_stops_before_wavelet_and_fit_and_redirects_outputs(self):
        sources = beta.notebook_prefix(beta.batch.DEFAULT_NOTEBOOK, "2022-01-01", "2022-01-02",
                                       Path("/tmp/beta_test"), Path("/tmp/local_cdf"))
        source = "\n".join(sources)
        ast.parse(source)
        self.assertNotIn("run_wavelet_segments(", source)
        self.assertNotIn("kappa_phase_summary_", source)
        self.assertNotIn("with ProcessPoolExecutor", source)
        self.assertNotIn(beta.batch.NOTEBOOK_OUTPUT_ROOT, source)
        self.assertIn("sm(beta_e_hfa)", source)
        self.assertIn("/tmp/local_cdf", source)
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(n, ast.Name) and n.id.startswith("v_sc") for n in ast.walk(tree)))
        self.assertNotIn("def transform_v_sc_to_dsi", source)
        self.assertIn("def rotate_vector_dsi_to_fac", source)
        self.assertIn("v_ion_fac = rotate_vector_dsi_to_fac", source)
        self.assertIn("v_sys_fac = v_ion_fac.rename('v_sys_fac')", source)

    def test_beta_is_computed_before_smoothing_not_as_ratio_of_smoothed_values(self):
        sources = beta.notebook_prefix(beta.batch.DEFAULT_NOTEBOOK, "2022-01-01", "2022-01-02",
                                       Path("/tmp/beta_test"), Path("/tmp/local_cdf"))
        definitions = []
        names = {"dedup_and_sort", "interp_unique_time", "rolling_mean_by_seconds",
                 "vector_component", "vector_perp_magnitude", "build_plasma_parameter_products"}
        for source in sources:
            definitions.extend(node for node in ast.parse(source).body
                               if isinstance(node, ast.FunctionDef) and node.name in names)
        namespace = {"np": np, "pd": pd, "xr": xr, "mu0": 4*np.pi*1e-7,
                     "elementary_charge": 1.602176634e-19, "proton_mass_kg": 1.6726219e-27,
                     "electron_mass_kg": 9.1093837e-31}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "plasma_test", "exec"), namespace)
        time = pd.date_range("2022-01-01", periods=5, freq="s")
        def da(values):
            return xr.DataArray(values, dims="time", coords={"time": time})
        namespace.update({"da_B64_dsi_clean_total": da([100.]*5),
                          "ND_electron_HFA": da([1., 10., 2., 8., 4.]),
                          "ND_electron_LEP": da([1.]*5),
                          "Temp_electron": da([100., 20., 50., 30., 100.]),
                          "Temp_ion": da([100., 2000., 500., 30., 1000.]),
                          "ion_mass": da([namespace["proton_mass_kg"]]*5)})
        vector = xr.DataArray(np.ones((5, 3)), dims=("time", "v_dim"),
                              coords={"time": time, "v_dim": [0, 1, 2]})
        namespace.update({"v_ion_fac": vector, "v_sys_fac": vector})
        ds, *_ = namespace["build_plasma_parameter_products"](vector.time, smoothing_sec=3.)
        raw = (2*namespace["mu0"] * namespace["ND_electron_HFA"] * 1e6
               * namespace["elementary_charge"] * namespace["Temp_electron"] / (100e-9)**2)
        expected = raw.rolling(time=3, center=True).mean().sel(time=ds.time)
        np.testing.assert_allclose(ds.electron_plasma_beta_HFA, expected, equal_nan=True)
        approximation = ds.ion_plasma_beta_HFA / ds["i-e_temp_ratio"]
        self.assertFalse(np.allclose(approximation.values, expected.values))

    def test_reference_mismatch_fails_and_nonfinite_samples_reduce_coverage(self):
        result = beta.describe([1., np.nan, 3.], beta.PREFIX)
        self.assertEqual(result[beta.PREFIX + "_n_total"], 3)
        self.assertEqual(result[beta.PREFIX + "_valid_fraction"], 2/3)
        self.assertEqual(result[beta.PREFIX + "_median"], 2.)
        beta.validate_reference({"n": 3.}, {"n": 3.})
        with self.assertRaises(ValueError):
            beta.validate_reference({"n": 3.}, {"n": 4.})

    def test_sidecar_requires_full_coverage_correct_master_and_unique_range(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "range_summary.csv"
            master = pd.DataFrame({"range_uid": ["a", "a", "b"], "phase_mode": ["all"]*3})
            sidecar = pd.DataFrame({"range_uid": ["a", "b"], "method": [beta.METHOD]*2,
                                    "status": ["complete"]*2,
                                    beta.PREFIX + "_median": [.1, .2],
                                    beta.PREFIX + "_valid_fraction": [1., .9]})
            def save(frame):
                frame.to_csv(path, index=False)
                path.with_suffix(".provenance.json").write_text(json.dumps({
                    "master_sha256": "master_hash", "method": beta.METHOD,
                    "summary_sha256": beta.batch.file_sha256(path)}))
            save(sidecar)
            merged = plotting.load_electron_beta(master, path, "master_hash")
            self.assertEqual(merged[beta.PREFIX + "_median"].tolist(), [.1, .1, .2])
            with self.assertRaises(ValueError):
                plotting.load_electron_beta(master, path, "wrong_hash")
            save(sidecar.iloc[:1])
            with self.assertRaises(ValueError):
                plotting.load_electron_beta(master, path, "master_hash")
            save(pd.concat([sidecar, sidecar.iloc[:1]]))
            with self.assertRaises(ValueError):
                plotting.load_electron_beta(master, path, "master_hash")


if __name__ == "__main__":
    unittest.main()
