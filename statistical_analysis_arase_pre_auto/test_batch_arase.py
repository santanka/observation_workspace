import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

import batch_arase as batch
import att_tolerant


class BatchAraseTests(unittest.TestCase):
    def test_event_orbit_cli_is_opt_in_and_supports_backfill(self):
        run_args = batch.build_parser().parse_args(["run"])
        self.assertFalse(run_args.plot_event_orbits)
        enabled = batch.build_parser().parse_args(["run", "--plot-event-orbits"])
        self.assertTrue(enabled.plot_event_orbits)
        backfill = batch.build_parser().parse_args([
            "event-orbits", "--range-id", "7", "--png-only"
        ])
        self.assertEqual(backfill.command, "event-orbits")
        self.assertEqual(backfill.range_id, [7])
        self.assertTrue(backfill.png_only)

    def test_manifest(self):
        payload, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        self.assertGreater(len(ranges), 0)
        self.assertEqual(len(ranges), len(payload["ranges"]))
        range_ids = [int(item["range_id"]) for item in ranges]
        self.assertEqual(range_ids, sorted(set(range_ids)))

    def test_notebook_transform_syntax(self):
        notebook = json.loads(batch.DEFAULT_NOTEBOOK.read_text())
        transformed = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = batch.transform_notebook_source(
                "".join(cell["source"]),
                "2022-09-01T05:46:33.183191",
                "2022-09-01T06:21:52.179714",
                Path("/tmp/arase-auto-test"),
            )
            ast.parse(source)
            transformed.append(source)
        ast.parse(batch.CONTEXT_CELL)
        ast.parse(batch.BATCH_PREAMBLE)
        joined = "\n".join(transformed)
        self.assertIn("PLOT_WAVELET_MEDIAN_PSD = False", joined)
        self.assertIn(batch.WAVELET_TEMP_DIR_SOURCE, joined)
        self.assertNotIn(batch.WAVELET_SAVE_DIR_SOURCE, joined)
        self.assertIn("max_gap='5min'", joined)
        self.assertIn("bootstrap_samples_path =", joined)
        self.assertIn("/tmp/arase-auto-test", joined)
        self.assertNotIn(batch.NOTEBOOK_OUTPUT_ROOT, joined)
        self.assertEqual(joined.count("no_update=True"), 10)
        self.assertIn("load_att_tolerant(time_range", joined)
        self.assertNotIn("psp.projects.erg.att(", joined)
        self.assertIn("WAVELET_N_JOBS = 1", joined)
        self.assertEqual(joined.count("n_cores = 1"), 2)
        self.assertEqual(joined.count("n_workers = 1"), 1)
        self.assertGreaterEqual(joined.count("n_jobs=1"), 2)
        self.assertNotIn("n_jobs=-1", joined)
        self.assertIn("ds_B_seg = batch_dedup_time(ds_B_seg)", joined)
        self.assertIn("v_ion_fac = batch_dedup_time(v_ion_fac)", joined)
        self.assertIn("nd_window = batch_rolling_window_samples", joined)
        self.assertIn("name_out='sundir_j2000'", joined)
        self.assertIn("noload=True", joined)
        for marker in batch.REQUIRED_NOTEBOOK_ANALYSIS_MARKERS:
            self.assertIn(marker, joined)

        required_transform_markers = (
            "max_gap='5min'",
            "bootstrap_samples_path =",
            "PLOT_WAVELET_MEDIAN_PSD = False",
            "WAVELET_N_JOBS = 1",
            "n_cores = 1",
            "n_workers = 1",
            "n_jobs=1",
            batch.WAVELET_TEMP_DIR_SOURCE,
            "load_att_tolerant(time_range",
            "return batch_dedup_time(da).sel",
            "ds_B_seg = batch_dedup_time(ds_B_seg)",
            "v_ion_fac = batch_dedup_time(v_ion_fac)",
            "nd_window = batch_rolling_window_samples",
            "no_update=True",
            "name_out='sundir_j2000'",
            "noload=True",
            *batch.REQUIRED_NOTEBOOK_ANALYSIS_MARKERS,
        )
        self.assertEqual(
            [marker for marker in required_transform_markers if marker not in joined], []
        )

    def test_execute_notebook_preflight_accepts_current_transform(self):
        args = SimpleNamespace(
            notebook=batch.DEFAULT_NOTEBOOK,
            start="2021-03-26T06:10:51.693925",
            end="2021-03-26T06:17:31.693925",
            output_root=Path("/tmp/arase-auto-preflight"),
            spedas_dir=Path("/mnt/j/observation_data"),
            kernel_name="python3",
        )
        with mock.patch("nbclient.NotebookClient.execute", return_value=None) as execute:
            self.assertEqual(batch.execute_notebook_once(args), 0)
        execute.assert_called_once_with()

    def test_batch_preamble_deduplicates_time_and_guards_empty_log_axis(self):
        namespace = {}
        exec(compile(batch.BATCH_PREAMBLE, "<batch-preamble>", "exec"), namespace)
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims="time",
            coords={"time": np.array([
                "2022-01-01T00:00:01", "2022-01-01T00:00:00",
                "2022-01-01T00:00:01",
            ], dtype="datetime64[ns]")},
        )
        deduped = namespace["batch_dedup_time"](da)
        self.assertEqual(deduped.sizes["time"], 2)
        self.assertTrue(np.all(np.diff(deduped.time.values) > np.timedelta64(0, "ns")))

        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = namespace["plt"].subplots()
            ax.plot([0, 1], [-2, -1])
            ax.set_yscale("log")
            fig.tight_layout()
            self.assertEqual(ax.get_yscale(), "linear")
            self.assertEqual(len(ax.texts), 1)
            path = Path(tmp) / "guarded.png"
            fig.savefig(path)
            namespace["plt"].close(fig)
            self.assertTrue(path.is_file())
            self.assertEqual(ax.get_yscale(), "linear")
            self.assertEqual(len(ax.texts), 1)

    def test_auto_kaw_additional_selection_semantics(self):
        notebook = json.loads(batch.DEFAULT_NOTEBOOK.read_text())
        assignment_names = {
            "E_SPIN_MIN_BAND_POINTS",
            "EB_SPIN_LOGRMSE_MAX_DEX",
            "EB_RESIDUAL_PEAK_MAX_DEX",
        }
        selected_nodes = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = batch.transform_notebook_source(
                "".join(cell["source"]), "a", "b", Path("/tmp/test")
            )
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in assignment_names
                    for target in node.targets
                ):
                    selected_nodes.append(node)
                elif (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "_additional_eb_selection_mask"
                ):
                    selected_nodes.append(node)

        namespace = {"np": np}
        module = ast.Module(body=selected_nodes, type_ignores=[])
        exec(compile(module, str(batch.DEFAULT_NOTEBOOK), "exec"), namespace)
        selection_mask = namespace["_additional_eb_selection_mask"]
        table = {
            "n_E_spinband_kaw": np.array([0, 9, 10, 10, 10, np.nan, 5, 10]),
            "logrmse_E_spinband_kaw": np.array([
                np.nan, np.nan, 0.70, 0.80, np.nan, 0.10, np.nan, 0.20,
            ]),
            "EB_residual_peak": np.array([
                0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.80, np.nan,
            ]),
        }
        expected = np.array([True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(selection_mask(table), expected)

    def test_auto_spinband_rmse_uses_kaw_frequency_mask(self):
        notebook = json.loads(batch.DEFAULT_NOTEBOOK.read_text())
        assignment_names = {
            "SPIN_FREQUENCY_HZ",
            "E_SPIN_FIT_RANGE_HZ",
            "E_SPIN_BAND_MULTIPLIERS",
            "E_SPIN_MIN_FIT_POINTS",
            "E_SPIN_MIN_BAND_POINTS",
        }
        selected_nodes = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = batch.transform_notebook_source(
                "".join(cell["source"]), "a", "b", Path("/tmp/test")
            )
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in assignment_names
                    for target in node.targets
                ):
                    selected_nodes.append(node)
                elif (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "_fit_e_spinband_baseline"
                ):
                    selected_nodes.append(node)

        namespace = {"np": np}
        module = ast.Module(body=selected_nodes, type_ignores=[])
        exec(compile(module, str(batch.DEFAULT_NOTEBOOK), "exec"), namespace)
        spinband_rmse = namespace["_fit_e_spinband_baseline"]

        freq = np.logspace(np.log10(0.01), np.log10(3.0), 512)
        spin = (freq >= 0.8 * 0.125) & (freq <= 6.0 * 0.125)
        kaw_frequency_mask = freq >= 0.30
        coherence = np.ones_like(freq)
        coherence_threshold = np.full_like(freq, 0.5)
        baseline = 4.0 * freq**-2.0

        outside_only = baseline.copy()
        outside_only[spin & ~kaw_frequency_mask] *= 10.0
        in_range = baseline.copy()
        in_range[spin & kaw_frequency_mask] *= 10.0

        outside_result = spinband_rmse(
            freq, outside_only, coherence, coherence_threshold, kaw_frequency_mask
        )
        in_range_result = spinband_rmse(
            freq, in_range, coherence, coherence_threshold, kaw_frequency_mask
        )
        self.assertAlmostEqual(outside_result[0], 0.0, places=12)
        self.assertAlmostEqual(in_range_result[0], 1.0, places=12)
        self.assertEqual(outside_result[5], int(np.count_nonzero(spin & kaw_frequency_mask)))

    def test_hfa_interpolation_strict_five_minute_boundary(self):
        notebook = json.loads(batch.DEFAULT_NOTEBOOK.read_text())
        source = batch.transform_notebook_source(
            "".join(notebook["cells"][61]["source"]), "a", "b", Path("/tmp/test")
        )
        namespace = {"np": np, "pd": pd, "xr": xr}
        exec(source, namespace)
        interpolate = namespace["interp_unique_time"]
        t0 = pd.Timestamp("2022-01-01")

        for gap, midpoint_is_finite in (
            (pd.Timedelta(minutes=4, seconds=59), True),
            (pd.Timedelta(minutes=5), False),
        ):
            times = [t0, t0 + gap / 2, t0 + gap]
            source_da = xr.DataArray(
                [1.0, 3.0], dims="time", coords={"time": [t0, t0 + gap]}
            )
            target = xr.DataArray(times, dims="time", coords={"time": times})
            result = interpolate(source_da, target, max_gap="5min").values
            self.assertEqual(bool(np.isfinite(result[1])), midpoint_is_finite)
            self.assertEqual(result[0], 1.0)
            self.assertEqual(result[2], 3.0)

    def test_empty_aggregate(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {
                "state_dir": Path(tmp) / "state",
                "output_root": Path(tmp) / "output",
            })()
            batch.aggregate(args, ranges[:2])
            status = pd.read_csv(Path(tmp) / "state" / "aggregate" / "range_status.csv")
            self.assertEqual(status["status"].tolist(), ["pending", "pending"])

    def test_default_paths_are_namespaced_by_manifest_content(self):
        ranges_path = Path("/tmp/example ranges.json")
        args = type("Args", (), {
            "ranges": ranges_path,
            "state_dir": None,
            "output_root": None,
        })()
        key = batch.resolve_run_paths(args, "0123456789abcdef")
        self.assertEqual(key, "example_ranges_0123456789ab")
        self.assertEqual(args.state_dir, batch.DEFAULT_STATE_ROOT / key)
        self.assertEqual(args.output_root, batch.DEFAULT_OUTPUT_ROOT / key)

    def test_product_download_ranges(self):
        tranges = batch.download_tranges(
            "2022-09-01T23:30:00", "2022-09-02T00:20:00"
        )
        self.assertEqual(
            tranges["exact"],
            ["2022-09-01T23:30:00", "2022-09-02T00:20:00"],
        )
        self.assertEqual(
            tranges["omni_1min"],
            ["2022-09-01T20:30:00", "2022-09-02T03:20:00"],
        )
        self.assertEqual(
            tranges["omni_hourly"], ["2022-09-01", "2022-09-03"]
        )

    def test_product_file_units_and_blocks(self):
        ranges = [
            {
                "range_id": 1,
                "start_time": "2022-09-30T23:30:00",
                "end_time": "2022-09-30T23:40:00",
            },
            {
                "range_id": 2,
                "start_time": "2022-09-30T23:50:00",
                "end_time": "2022-10-01T00:20:00",
            },
        ]

        mgf_units = batch.build_download_units(ranges, "mgf_l2_64hz")
        self.assertEqual([unit.key for unit in mgf_units], ["20220930T23", "20221001T00"])
        self.assertEqual(mgf_units[0].range_ids, (1, 2))
        self.assertEqual(mgf_units[1].range_ids, (2,))
        mgf_blocks = batch.block_download_units(mgf_units)
        self.assertEqual(len(mgf_blocks), 1)
        self.assertEqual(mgf_blocks[0].request_end.isoformat(), "2022-10-01T00:59:59")

        daily_units = batch.build_download_units(ranges, "pwe_efd_l2_64")
        self.assertEqual([unit.key for unit in daily_units], ["20220930", "20221001"])
        self.assertEqual(len(batch.block_download_units(daily_units)), 1)

        att_units = batch.build_download_units([{
            "range_id": 3,
            "start_time": "2022-10-01T00:00:30",
            "end_time": "2022-10-01T00:01:30",
        }], "att_l2")
        self.assertEqual([unit.key for unit in att_units], ["20220930", "20221001"])
        self.assertTrue(all(unit.range_ids == (3,) for unit in att_units))

        omni_months = batch.build_download_units(ranges, "omni_1min")
        self.assertEqual([unit.key for unit in omni_months], ["202209", "202210"])
        self.assertTrue(all(
            block.request_end - block.start == pd.Timedelta(days=1, seconds=-1)
            for block in batch.block_download_units(omni_months)
        ))
        omni_halves = batch.build_download_units(ranges, "omni_hourly")
        self.assertEqual([unit.key for unit in omni_halves], ["2022H2"])

    def test_download_block_cap_and_plan_reduction(self):
        ranges = [
            {
                "range_id": i + 1,
                "start_time": f"2022-09-01T0{i}:10:00",
                "end_time": f"2022-09-01T0{i}:20:00",
            }
            for i in range(3)
        ]
        units = batch.build_download_units(ranges, "mgf_l2_64hz")
        blocks = batch.block_download_units(units, max_block_units=2)
        self.assertEqual([len(block.units) for block in blocks], [2, 1])

        _, current_ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        plan = batch.build_download_plan(current_ranges)
        old_request_count = len(current_ranges) * len(batch.PRODUCTS)
        self.assertLess(len(plan), old_request_count)
        self.assertTrue(all(block.units for block in plan))

    def test_download_unit_marker_requires_existing_files(self):
        unit = batch.DownloadUnit(
            product="mgf_l2_64hz",
            key="20220901T00",
            start=pd.Timestamp("2022-09-01T00:00:00").to_pydatetime(),
            end_exclusive=pd.Timestamp("2022-09-01T01:00:00").to_pydatetime(),
            range_ids=(1,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            local_file = Path(tmp) / "erg_mgf_test.cdf"
            local_file.write_bytes(b"cdf")
            marker_path = batch.download_unit_marker(state_dir, unit)
            batch.atomic_json(marker_path, {
                "status": "complete",
                "product": unit.product,
                "unit_key": unit.key,
                "unit_start": unit.start.isoformat(),
                "unit_end_exclusive": unit.end_exclusive.isoformat(),
                "files": [str(local_file)],
            })
            self.assertTrue(batch.compatible_download_unit(marker_path, unit))
            local_file.unlink()
            self.assertFalse(batch.compatible_download_unit(marker_path, unit))

    def test_download_files_are_assigned_to_their_physical_unit(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2022-09-01T00:10:00",
            "end_time": "2022-09-01T01:10:00",
        }]
        units = batch.build_download_units(ranges, "mgf_l2_64hz")
        files = [
            "/data/erg_mgf_l2_64hz_dsi_2022090100_v01.cdf",
            "/data/erg_mgf_l2_64hz_dsi_2022090101_v01.cdf",
        ]
        self.assertEqual(batch._files_for_unit(files, units[0]), [files[0]])
        self.assertEqual(batch._files_for_unit(files, units[1]), [files[1]])

    def test_att_local_files_must_cover_every_requested_day(self):
        files = [
            "/data/erg_att_l2_20220901_v03.txt",
            "/data/erg_att_l2_20220902_v03.txt",
        ]
        self.assertTrue(batch._att_files_cover_trange(
            files, "2022-09-01T00:00:00", "2022-09-02T23:59:59"
        ))
        self.assertFalse(batch._att_files_cover_trange(
            files[:1], "2022-09-01T00:00:00", "2022-09-02T23:59:59"
        ))

    def test_mgf_prefetch_uses_complete_local_hour_coverage(self):
        files = [
            "/data/erg_mgf_l2_64hz_dsi_2022090100_v01.cdf",
            "/data/erg_mgf_l2_64hz_dsi_2022090101_v01.cdf",
        ]
        loader = mock.Mock(return_value=files)
        result = batch._prefetch_mgf_files(
            loader, ["2022-09-01T00:00:00", "2022-09-01T01:59:59"]
        )
        self.assertEqual(result, files)
        loader.assert_called_once()
        self.assertTrue(loader.call_args.kwargs["no_update"])

    def test_mgf_prefetch_downloads_only_missing_hours(self):
        local = ["/data/erg_mgf_l2_64hz_dsi_2022090100_v01.cdf"]
        downloaded = ["/data/erg_mgf_l2_64hz_dsi_2022090101_v01.cdf"]
        loader = mock.Mock(side_effect=[local, downloaded])
        result = batch._prefetch_mgf_files(
            loader, ["2022-09-01T00:00:00", "2022-09-01T01:59:59"]
        )
        self.assertEqual(result, local + downloaded)
        self.assertEqual(loader.call_count, 2)
        self.assertTrue(loader.call_args_list[0].kwargs["no_update"])
        self.assertNotIn("no_update", loader.call_args_list[1].kwargs)

    def test_mgf_prefetch_rejects_incomplete_combined_coverage(self):
        local = ["/data/erg_mgf_l2_64hz_dsi_2022090100_v01.cdf"]
        loader = mock.Mock(side_effect=[local, []])
        with self.assertRaisesRegex(RuntimeError, "2022090101"):
            batch._prefetch_mgf_files(
                loader, ["2022-09-01T00:00:00", "2022-09-01T01:59:59"]
            )

    def test_omni_hourly_request_uses_date_only(self):
        self.assertEqual(
            batch._product_request_trange(
                "omni_hourly", "2022-07-01T00:00:00", "2022-07-01T23:59:59"
            ),
            ["2022-07-01", "2022-07-01"],
        )

    def test_prefetch_block_writes_reusable_unit_markers(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2022-09-01T00:10:00",
            "end_time": "2022-09-01T01:10:00",
        }]
        units = batch.build_download_units(ranges, "mgf_l2_64hz")
        block = batch.block_download_units(units)[0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = type("Args", (), {
                "spedas_dir": root / "spedas",
                "download_timeout": 30,
            })()

            def fake_run_child(command, log_path, timeout, env):
                result_path = Path(command[command.index("--result-json") + 1])
                local_files = [
                    root / "spedas" / f"erg_mgf_l2_64hz_dsi_{token}_v01.cdf"
                    for token in ("2022090100", "2022090101")
                ]
                for local_file in local_files:
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    local_file.write_bytes(b"cdf")
                batch.atomic_json(
                    result_path, {"files": [str(path) for path in local_files]},
                )
                return 0

            with mock.patch.object(batch, "run_child", side_effect=fake_run_child):
                _, failed_units = batch._run_prefetch_block(
                    args, block, root / "state", {},
                )

            self.assertEqual(failed_units, ())
            for unit in units:
                marker_path = batch.download_unit_marker(root / "state", unit)
                self.assertTrue(batch.compatible_download_unit(marker_path, unit))

    def test_prefetch_block_only_fails_the_missing_unit(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2022-09-01T00:10:00",
            "end_time": "2022-09-01T01:10:00",
        }]
        units = batch.build_download_units(ranges, "mgf_l2_64hz")
        block = batch.block_download_units(units)[0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = type("Args", (), {
                "spedas_dir": root / "spedas",
                "download_timeout": 30,
            })()

            def fake_run_child(command, log_path, timeout, env):
                result_path = Path(command[command.index("--result-json") + 1])
                local_file = (
                    root / "spedas" / "erg_mgf_l2_64hz_dsi_2022090100_v01.cdf"
                )
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_bytes(b"cdf")
                batch.atomic_json(result_path, {"files": [str(local_file)]})
                return 0

            with mock.patch.object(batch, "run_child", side_effect=fake_run_child):
                _, failed_units = batch._run_prefetch_block(
                    args, block, root / "state", {},
                )

            self.assertEqual(failed_units, (units[1],))
            self.assertTrue(batch.compatible_download_unit(
                batch.download_unit_marker(root / "state", units[0]), units[0]
            ))
            missing = json.loads(
                batch.download_unit_marker(root / "state", units[1]).read_text()
            )
            self.assertEqual(missing["status"], "missing_remote")

    def test_known_missing_unit_is_reused_without_download(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2022-09-01T00:10:00",
            "end_time": "2022-09-01T00:20:00",
        }]
        unit = batch.build_download_units(ranges, "mgf_l2_64hz")[0]
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            marker = batch.download_unit_marker(state_dir, unit)
            batch._write_unit_status(
                state_dir, unit, "missing_remote",
                batch.block_download_units([unit])[0],
                files=[],
            )
            self.assertTrue(batch.known_missing_download_unit(marker, unit))
            args = type("Args", (), {
                "state_dir": state_dir,
                "spedas_dir": Path(tmp) / "spedas",
                "force": False,
                "download_workers": 2,
                "download_timeout": 30,
            })()
            with (
                mock.patch.object(batch, "PRODUCTS", ("mgf_l2_64hz",)),
                mock.patch.object(batch, "_run_prefetch_block") as worker,
            ):
                self.assertEqual(batch.run_prefetch(args, ranges), {1})
            worker.assert_not_called()


    def test_att_failure_classification_is_narrow(self):
        att_log = (
            "erg_interpolate_att\n"
            "Downloading remote index: https://example/erg/att/txt/\n"
            "ReadTimeoutError: read timed out\n"
            "Connection error getting remote index\n"
            "ValueError: No objects to concatenate\n"
        )
        self.assertEqual(
            batch.classify_range_failure(att_log, 1),
            ("att_download_transient", True),
        )
        self.assertEqual(
            batch.classify_range_failure(
                "ValueError: No overlapping E64/B64 DSI segments", 1
            ),
            ("no_overlapping_efd_mgf_waveform", False),
        )
        self.assertEqual(
            batch.classify_range_failure(
                "erg_interpolate_att\nValueError: No objects to concatenate", 1
            ),
            ("att_data_unavailable", False),
        )
        self.assertEqual(
            batch.classify_range_failure("nbclient.exceptions.DeadKernelError: Kernel died", 1),
            ("dead_kernel_resource", True),
        )
        self.assertEqual(
            batch.classify_range_failure(
                "erg_att_l2\nValueError: could not convert string to float", 1
            ),
            ("invalid_att_format", False),
        )
        self.assertEqual(
            batch.classify_range_failure(
                "RuntimeError: Notebook batch transformation incomplete: ['marker']", 1
            ),
            ("batch_transform_incomplete", False),
        )

    def test_att_missing_delimiter_repair_is_narrow(self):
        broken = "135.2657   26.2539-151515.3574       0.0"
        repaired = att_tolerant.repair_missing_att_delimiter(broken)
        self.assertEqual(
            repaired.split(), ["135.2657", "26.2539", "-151515.3574", "0.0"]
        )
        self.assertEqual(
            att_tolerant.repair_missing_att_delimiter("1.0  -2.0  3.0"),
            "1.0  -2.0  3.0",
        )

    def test_att_parser_preserves_nan_and_rejects_invalid_numeric_tokens(self):
        header = "ATT header\n" + "metadata\n" * 10
        valid = "2020-03-01/00:00:00.0000 " + " ".join(["1.0000"] * 13)
        fields = ["2.0000"] * 13
        fields[8] = "NaN"  # column 9: spin phase
        missing = "2020-03-01/00:00:08.0000 " + " ".join(fields)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "att.txt"
            path.write_text(header + valid + "\n" + missing + "\n")
            parsed = att_tolerant.parse_att_files([path], store=False)
            phase = parsed["erg_att_spphase"]["y"]
            self.assertEqual(phase[0], 1.0)
            self.assertTrue(np.isnan(phase[1]))
            self.assertEqual(len(parsed["erg_att_spphase"]["x"]), 2)
            att_tolerant.validate_att_files([path])
            path.write_text(header + valid + "\n" + missing.replace("NaN", "BAD_TOKEN") + "\n")
            with self.assertRaises(ValueError):
                att_tolerant.parse_att_files([path], store=False)

    def test_retryable_download_ranges_exclude_missing_remote(self):
        ranges = [{
            "range_id": 1,
            "start_time": "2022-09-01T12:10:00",
            "end_time": "2022-09-01T12:20:00",
        }]
        unit = batch.build_download_units(ranges, "att_l2")[0]
        block = batch.block_download_units([unit])[0]
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            batch._write_unit_status(
                state_dir, unit, "failed", block, files=[], returncode=1
            )
            self.assertEqual(
                batch.retryable_download_range_ids(state_dir, ranges), {1}
            )
            batch._write_unit_status(
                state_dir, unit, "missing_remote", block, files=[]
            )
            self.assertEqual(
                batch.retryable_download_range_ids(state_dir, ranges), set()
            )

    def test_run_ranges_marks_att_network_failure_retryable(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        item = ranges[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = type("Args", (), {
                "notebook": batch.DEFAULT_NOTEBOOK,
                "state_dir": root / "state",
                "output_root": root / "output",
                "spedas_dir": root / "spedas",
                "kernel_name": "python3",
                "analysis_timeout": 30,
                "force": False,
            })()

            wavelet_temp_dirs = []

            def fake_run_child(command, log_path, timeout, env):
                wavelet_temp_dir = Path(env["ARASE_WAVELET_TEMP_DIR"])
                self.assertTrue(wavelet_temp_dir.is_dir())
                (wavelet_temp_dir / "temporary.nc").write_bytes(b"netcdf")
                wavelet_temp_dirs.append(wavelet_temp_dir)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "erg_interpolate_att\n"
                    "https://example/erg/att/txt/\n"
                    "ReadTimeoutError\n"
                    "Connection error getting remote index\n"
                )
                return 1

            with mock.patch.object(batch, "run_child", side_effect=fake_run_child):
                batch.run_ranges(args, [item], "ranges-hash")

            status = json.loads(batch.status_path(args.state_dir, item).read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["failure_class"], "att_download_transient")
            self.assertTrue(status["retryable"])
            self.assertEqual(status["attempt"], 1)
            self.assertEqual(status["wavelet_cache_policy"], "temporary")
            self.assertTrue(status["wavelet_temp_removed"])
            self.assertEqual(len(wavelet_temp_dirs), 1)
            self.assertFalse(wavelet_temp_dirs[0].exists())

    def test_run_ranges_removes_wavelet_temp_after_complete_status(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        item = ranges[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = type("Args", (), {
                "notebook": batch.DEFAULT_NOTEBOOK,
                "state_dir": root / "state",
                "output_root": root / "output",
                "spedas_dir": root / "spedas",
                "wavelet_scratch_root": root / "wavelet-scratch",
                "kernel_name": "python3",
                "analysis_timeout": 30,
                "force": False,
            })()
            wavelet_temp_dirs = []

            def fake_run_child(command, log_path, timeout, env):
                wavelet_temp_dir = Path(env["ARASE_WAVELET_TEMP_DIR"])
                (wavelet_temp_dir / "temporary.nc").write_bytes(b"netcdf")
                wavelet_temp_dirs.append(wavelet_temp_dir)
                summary = batch.expected_summary(item, args.output_root)
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text("phase_mode,kappa_E\nall,1.0\n")
                return 0

            with mock.patch.object(batch, "run_child", side_effect=fake_run_child):
                batch.run_ranges(args, [item], "ranges-hash")

            status = json.loads(batch.status_path(args.state_dir, item).read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["wavelet_cache_policy"], "temporary")
            self.assertTrue(status["wavelet_temp_removed"])
            self.assertEqual(len(wavelet_temp_dirs), 1)
            self.assertFalse(wavelet_temp_dirs[0].exists())

    def test_run_ranges_marks_empty_input_as_terminal_excluded(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        item = ranges[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = type("Args", (), {
                "notebook": batch.DEFAULT_NOTEBOOK,
                "state_dir": root / "state",
                "output_root": root / "output",
                "spedas_dir": root / "spedas",
                "wavelet_scratch_root": root / "wavelet-scratch",
                "kernel_name": "python3",
                "analysis_timeout": 30,
                "force": False,
            })()

            def fake_run_child(command, log_path, timeout, env):
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "ValueError: zero-size array to reduction operation minimum"
                )
                return 1

            with mock.patch.object(batch, "run_child", side_effect=fake_run_child):
                batch.run_ranges(args, [item], "ranges-hash")

            status = json.loads(batch.status_path(args.state_dir, item).read_text())
            self.assertEqual(status["status"], "excluded")
            self.assertEqual(status["failure_class"], "empty_input_data")
            self.assertFalse(status["retryable"])
            self.assertNotIn(item, batch.noncomplete_ranges(args.state_dir, [item]))

    def test_postpass_retries_only_retryable_analysis_ranges(self):
        ranges = [
            {"range_id": 1, "start_time": "2022-09-01T00:00:00", "end_time": "2022-09-01T00:10:00"},
            {"range_id": 2, "start_time": "2022-09-01T01:00:00", "end_time": "2022-09-01T01:10:00"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            batch.atomic_json(batch.status_path(state_dir, ranges[0]), {
                "status": "failed", "retryable": True,
                "failure_class": "dead_kernel_resource", "auto_retry_count": 0,
            })
            batch.atomic_json(batch.status_path(state_dir, ranges[1]), {
                "status": "failed", "retryable": False,
            })
            args = type("Args", (), {
                "state_dir": state_dir,
                "postpass_retries": 1,
            })()
            with (
                mock.patch.object(batch, "retryable_download_range_ids", return_value=set()),
                mock.patch.object(batch, "run_ranges") as run_ranges,
                mock.patch.object(batch.time, "sleep") as sleep,
            ):
                batch.run_postpass_retries(args, ranges, "ranges-hash")
            sleep.assert_called_once_with(batch.POSTPASS_RETRY_DELAYS_SEC[0])
            run_ranges.assert_called_once()
            retry_args, retry_ranges, retry_hash = run_ranges.call_args.args
            self.assertTrue(retry_args._auto_retry)
            self.assertEqual(retry_ranges, [ranges[0]])
            self.assertEqual(retry_hash, "ranges-hash")

    def test_analysis_auto_retry_limit_and_noncomplete_selection(self):
        ranges = [
            {"range_id": 1}, {"range_id": 2}, {"range_id": 3}, {"range_id": 4},
            {"range_id": 5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            batch.atomic_json(batch.status_path(state_dir, ranges[0]), {
                "status": "complete",
            })
            batch.atomic_json(batch.status_path(state_dir, ranges[1]), {
                "status": "failed", "retryable": True,
                "failure_class": "dead_kernel_resource", "auto_retry_count": 0,
            })
            batch.atomic_json(batch.status_path(state_dir, ranges[2]), {
                "status": "failed", "retryable": True,
                "failure_class": "dead_kernel_resource", "auto_retry_count": 1,
            })
            batch.atomic_json(batch.status_path(state_dir, ranges[4]), {
                "status": "excluded", "failure_class": "empty_input_data",
            })
            self.assertEqual(
                [r["range_id"] for r in batch.retryable_analysis_ranges(state_dir, ranges)],
                [2],
            )
            self.assertEqual(
                [r["range_id"] for r in batch.noncomplete_ranges(state_dir, ranges)],
                [2, 3, 4],
            )

    def test_download_worker_default(self):
        args = batch.build_parser().parse_args(["prefetch"])
        self.assertEqual(args.download_workers, batch.DEFAULT_DOWNLOAD_WORKERS)
        run_args = batch.build_parser().parse_args(["run"])
        self.assertEqual(run_args.postpass_retries, batch.POSTPASS_RETRIES)
        self.assertFalse(run_args.noncomplete_only)

    def test_aggregate_excludes_unaccepted_old_summary(self):
        _, ranges = batch.load_ranges(batch.DEFAULT_RANGES)
        item = ranges[0]
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            output_root = Path(tmp) / "output"
            summary = batch.expected_summary(item, output_root)
            summary.parent.mkdir(parents=True)
            pd.DataFrame({"phase": ["old"]}).to_csv(summary, index=False)
            args = type("Args", (), {
                "state_dir": state_dir,
                "output_root": output_root,
            })()
            batch.aggregate(args, [item])
            master = state_dir / "aggregate" / "arase_phase_master.csv"
            self.assertFalse(master.exists())


if __name__ == "__main__":
    unittest.main()
