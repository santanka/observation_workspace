import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import prepare_electron_beta_statistics_parallel as parallel
import run_environment_statistics_parallel as runner
from statistics_period import resolve_period


FAKE_WORKER = '''import json, os, pathlib, sys, time
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
start = time.time()
time.sleep(request.get("delay", .3))
mode = request.get("mode", "ok")
if mode == "fatal":
    print("RuntimeError: Offline beta_e reconstruction: network request blocked", flush=True)
    sys.exit(2)
if mode == "failed":
    sys.exit(3)
result = {"range_uid": request["range_uid"], "signature": request["signature"],
          "status": "complete", "start": start, "finish": time.time(),
          "pid": os.getpid(), "omp": os.environ.get("OMP_NUM_THREADS")}
pathlib.Path(request["result"]).write_text(json.dumps(result))
'''


def requests_at(root, count):
    return [{"range_uid": f"range:{i}", "signature": f"original-signature-{i}",
             "result": str(root / f"result_{i}.json")} for i in range(count)]


class ParallelBetaTests(unittest.TestCase):
    def test_legacy_orphan_worker_and_parent_prevent_shared_output_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "environment_inputs/beta_e"
            proc = root / "proc"
            for pid, options in (("123001", ["--worker-request", str(output / "ranges/a.request.json")]),
                                 ("123002", ["--year", "2022", "--statistics-dir", str(root)]),
                                 ("123003", ["--worker-request", str(root / "other/ranges/a.request.json")])):
                directory = proc / pid
                directory.mkdir(parents=True)
                command = [parallel.sys.executable, str(Path(parallel.serial.__file__).resolve()), *options]
                (directory / "cmdline").write_bytes(b"\0".join(os.fsencode(a) for a in command) + b"\0")
            self.assertEqual(sorted(parallel.legacy_processes(output, proc)), [123001, 123002])

    def worker_patch(self, root):
        script = root / "fake_worker.py"
        script.write_text(FAKE_WORKER)
        return mock.patch.object(parallel, "worker_command",
                                 side_effect=lambda path: [parallel.sys.executable, str(script), str(path)])

    def test_real_child_processes_overlap_and_compatible_results_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = requests_at(root, 5)
            cached = Path(requests[0]["result"])
            cached.write_text(json.dumps({**requests[0], "status": "complete", "legacy": True}))
            original = cached.read_bytes()
            with self.worker_patch(root):
                report = parallel.run_requests(requests, 2, 10, root, root)
            self.assertEqual(report["cached_complete"], 1)
            self.assertEqual(report["new_complete"], 4)
            self.assertEqual(cached.read_bytes(), original)
            results = [parallel.read_complete(r) for r in requests[1:]]
            events = sorted([(r["start"], 1) for r in results] + [(r["finish"], -1) for r in results])
            active, maximum = 0, 0
            for _, delta in events:
                active += delta
                maximum = max(maximum, active)
            self.assertEqual(maximum, 2)
            self.assertEqual(len({r["pid"] for r in results}), 4)
            self.assertTrue(all(r["omp"] == "1" for r in results))

    def test_fatal_offline_error_stops_running_children_and_new_submissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = requests_at(root, 3)
            requests[0].update(mode="fatal", delay=.1)
            requests[1]["delay"] = 5.
            sentinel = root / "range_summary.csv"
            sentinel.write_text("previous verified aggregate")
            with self.worker_patch(root), self.assertRaisesRegex(RuntimeError, "Offline/source-contract"):
                parallel.run_requests(requests, 2, 10, root, root)
            self.assertFalse(Path(requests[1]["result"]).exists())
            self.assertFalse(Path(requests[2]["result"]).with_suffix(".request.json").exists())
            self.assertEqual(sentinel.read_text(), "previous verified aggregate")
            report = json.loads((root / "parallel_execution.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["failed"]), 1)

    def test_timeout_is_recorded_and_does_not_create_a_success_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = requests_at(root, 1)
            requests[0]["delay"] = 5.
            with self.worker_patch(root), self.assertRaisesRegex(RuntimeError, "incomplete"):
                parallel.run_requests(requests, 1, .1, root, root)
            self.assertIsNone(parallel.read_complete(requests[0]))
            self.assertEqual(json.loads((root / "failures.json").read_text())[0]["returncode"], 124)

    def test_scheduling_configuration_does_not_enter_scientific_worker_arguments(self):
        args = parallel.build_parser().parse_args(["--year", "2022", "--workers", "4",
                                                   "--range-uid", "a", "--range-uid", "b"])
        options = parallel.serial_arguments(args)
        self.assertNotIn("--workers", options)
        restored = parallel.serial.build_parser().parse_args(options)
        self.assertEqual(restored.range_uid, ["a", "b"])
        self.assertEqual(restored.year, 2022)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parallel.build_parser().parse_args(["--year", "2022", "--workers", "0"])

    def test_complete_cache_is_aggregated_with_original_code_and_parallel_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "environment_inputs/beta_e"
            output.mkdir(parents=True)
            requests = requests_at(output, 2)
            for request in requests:
                Path(request["result"]).write_text(json.dumps({**request, "status": "complete"}))
            def serial_main(options):
                if "--plan-only" in options:
                    (output / "plan.json").write_text(json.dumps(requests))
                else:
                    (output / "range_summary.provenance.json").write_text(json.dumps({"method": parallel.serial.METHOD}))
                return 0
            original_child = parallel.serial.batch.run_child
            with mock.patch.object(parallel.serial, "main", side_effect=serial_main) as base, \
                 mock.patch.object(parallel, "worker_command", side_effect=AssertionError("cached")):
                self.assertEqual(parallel.main(["--year", "2022", "--statistics-dir", str(root), "--workers", "2"]), 0)
            self.assertEqual(base.call_count, 2)
            self.assertIs(parallel.serial.batch.run_child, original_child)
            metadata = json.loads((output / "range_summary.provenance.json").read_text())
            self.assertEqual(metadata["parallel_execution"]["workers"], 2)


class ParallelPipelineTests(unittest.TestCase):
    def test_pipeline_passes_workers_only_to_beta_preparation(self):
        args = runner.build_parser().parse_args(["--year", "2022", "--workers", "2"])
        steps = runner.commands(args, Path("/tmp/prepared"), resolve_period(args))
        beta = next(command for name, command in steps if name == "prepare_beta_e")
        self.assertTrue(beta[1].endswith("prepare_electron_beta_statistics_parallel.py"))
        self.assertEqual(beta[beta.index("--workers")+1], "2")
        self.assertTrue(all("--workers" not in command for name, command in steps if name != "prepare_beta_e"))
        self.assertNotIn("--min-valid-fraction", steps[-1][1])

    def test_entrypoint_restores_legacy_module_after_failure(self):
        original = runner.pipeline.build_parser, runner.pipeline.commands, runner.pipeline.__file__
        with mock.patch.object(runner.pipeline, "main", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                runner.main(["--year", "2022", "--workers", "2"])
        self.assertEqual((runner.pipeline.build_parser, runner.pipeline.commands, runner.pipeline.__file__), original)


if __name__ == "__main__":
    unittest.main()
