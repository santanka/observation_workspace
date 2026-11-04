from argparse import Namespace
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

import prepare_system_velocity_statistics as velocity
import plot_environment_parameter_statistics as environment
import run_statistics as unified
from test_environment_parameter_statistics import make_frame


def complete(request):
    return {**velocity.plasma.describe([1., 2., 3.], velocity.PREFIX),
            'range_uid': request['range_uid'], 'signature': request['signature'],
            'method': velocity.METHOD, 'status': 'complete', 'reference_check': 'passed'}


class VelocityTests(unittest.TestCase):
    def namespace(self):
        times = pd.date_range('2022-01-01', periods=3, freq='s')
        return {'ds_velocity_ms_toroidal': xr.Dataset({'perp_sys_speed': ('time', [-2., 0., 2.])}, coords={'time': times}),
                'ds_velocity_ms_poloidal': xr.Dataset({'perp_sys_speed': ('time', [3., -1., 2.])}, coords={'time': times})}

    def test_y_component_uses_saved_grid_and_x_reference_not_vector_magnitude(self):
        namespace = self.namespace()
        times = namespace['ds_velocity_ms_toroidal'].time.values
        reference = velocity.plasma.describe([-2., 0., 2.], velocity.X_PREFIX)
        result, rebuilt = velocity.summarize_velocity(namespace, times, reference)
        self.assertEqual(result[velocity.PREFIX + '_median'], 2.)
        self.assertEqual(result[velocity.PREFIX + '_n_valid'], 3)
        self.assertEqual(rebuilt, reference)
        bad = dict(reference, **{velocity.X_PREFIX + '_median': 100.})
        with self.assertRaisesRegex(ValueError, 'differs'):
            velocity.summarize_velocity(namespace, times, bad)
        bad = dict(reference, **{velocity.X_PREFIX + '_n_total': 4})
        with self.assertRaisesRegex(ValueError, 'sample count'):
            velocity.summarize_velocity(namespace, times, bad)

    def test_worker_checksum_reference_cache_and_network_guard_without_cdf_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = root / 'notebook.json'
            notebook.write_text('{}')
            timeseries = root / 'times.csv'
            pd.DataFrame({'time': pd.date_range('2022-01-01', periods=3, freq='s')}).to_csv(timeseries, index=False)
            request = {'range_uid': 'a', 'signature': 'signature', 'start': '2022-01-01T00:00:00',
                       'end': '2022-01-01T00:00:02', 'allow_network': False,
                       'notebook': str(notebook), 'timeseries': str(timeseries),
                       'result': str(root / 'result.json'),
                       'reference': velocity.plasma.describe([-2., 0., 2.], velocity.X_PREFIX)}
            request.update({k: velocity.batch.file_sha256(p) for k, p in velocity.source_paths(request).items()})
            path = root / 'request.json'
            path.write_text(json.dumps(request))
            def execute(_request, _scratch):
                self.assertIn(str(velocity.batch.ROOT), sys.path)
                return self.namespace()
            script_only_path = [s for s in sys.path if Path(s or '.').resolve() != velocity.batch.ROOT]
            with mock.patch.object(sys, 'path', script_only_path), mock.patch.object(velocity, 'execute_prefix', side_effect=execute):
                velocity.worker(path)
            self.assertIsNotNone(velocity.read_complete(request))
            timeseries.write_text('time\n2022-01-02\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                velocity.worker(path)
        original = socket.getaddrinfo
        with velocity.offline_network(True):
            with self.assertRaisesRegex(RuntimeError, 'network request blocked'):
                socket.getaddrinfo('never-contact.example', 80)
        self.assertIs(socket.getaddrinfo, original)

    def test_authoritative_prefix_stops_before_cwt_and_retains_smoothed_y(self):
        request = {'notebook': str(velocity.batch.DEFAULT_NOTEBOOK), 'start': '2022-09-01T00:00:00',
                   'end': '2022-09-01T00:05:00', 'spedas_dir': str(velocity.batch.DEFAULT_SPEDAS_DIR)}
        sources = velocity.prefix_sources(request, Path('/tmp/unused_vy_prefix_check'))
        source = '\n'.join(sources)
        self.assertIn("'perp_sys_speed': sm(vsys_y)", source)
        self.assertNotIn('run_wavelet_segments(', source)
        self.assertNotIn('bootstrap_samples_path =', source)

    def test_plan_hashes_common_sources_once_and_network_policy_does_not_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master = make_frame()
            master['range_uid'] = [f'uid-{i}' for i in range(len(master))]
            master['system_velocity_policy'] = 'observed_partial_ion_moment_no_spacecraft_subtraction'
            master['range_start'], master['range_end'] = '2022-01-01T00:00:00', '2022-01-01T00:00:02'
            paths = []
            for i in range(len(master)):
                directory = root / str(i) / 'EB_ratio_toroidal_eachtime'
                directory.mkdir(parents=True)
                paths.append(str(directory / 'kappa_phase_summary_toroidal.csv'))
                (directory / 'kappa_timeseries_toroidal.csv').write_text('time\n2022-01-01T00:00:00\n')
            master['source_summary'] = paths
            reference = velocity.plasma.describe([1., 2., 3.], velocity.X_PREFIX)
            for column, value in reference.items():
                master[column] = value
            args = velocity.build_parser().parse_args(['--year', '2022'])
            with mock.patch.object(velocity, 'prefix_sources', return_value=[]), \
                 mock.patch.object(velocity.batch, 'file_sha256', wraps=velocity.batch.file_sha256) as hashed:
                requests = velocity.build_requests(args, master, {'annual_master_sha256': 'master'}, root)
                self.assertEqual(hashed.call_count, 4 + len(master))
            args.allow_network = True
            with mock.patch.object(velocity, 'prefix_sources', return_value=[]):
                network = velocity.build_requests(args, master, {'annual_master_sha256': 'master'}, root)
            self.assertEqual([r['signature'] for r in requests], [r['signature'] for r in network])

    def test_parallel_scheduler_reuses_success_and_retries_only_failed_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = [{'range_uid': str(i), 'signature': str(i), 'result': str(root / f'{i}.json')} for i in range(5)]
            velocity.atomic_json(Path(requests[0]['result']), complete(requests[0]))
            args = Namespace(workers=2)
            active, maximum, called = 0, 0, []
            lock = threading.Lock()
            def execute(_pool, request):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    called.append(request['range_uid'])
                time.sleep(.02)
                velocity.atomic_json(Path(request['result']), complete(request))
                with lock:
                    active -= 1
                return {'range_uid': request['range_uid'], 'status': 'complete'}
            with mock.patch.object(velocity.RangePool, 'execute', execute):
                velocity.run_requests(requests, args, root)
            self.assertEqual(set(called), {'1', '2', '3', '4'})
            self.assertLessEqual(maximum, 2)
            with mock.patch.object(velocity.RangePool, 'execute', side_effect=AssertionError('cache reuse')):
                velocity.run_requests(requests, args, root)
            Path(requests[3]['result']).unlink()
            called.clear()
            with mock.patch.object(velocity.RangePool, 'execute', execute):
                velocity.run_requests(requests, args, root)
            self.assertEqual(called, ['3'])

    def test_pool_passes_coordinator_lock_and_fatal_failure_stops_scheduling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {'range_uid': 'uid-a', 'signature': 'a', 'result': str(root / 'a.json')}
            velocity.atomic_json(Path(request['result']), complete(request))
            args = Namespace(workers=1, timeout=10, spedas_dir=root)
            process = mock.Mock()
            process.wait.return_value = 0
            with mock.patch.object(velocity.subprocess, 'Popen', return_value=process) as started:
                outcome = velocity.RangePool(args, coordinator_lock_fd=99).execute(request)
            self.assertEqual(outcome['status'], 'complete')
            self.assertEqual(started.call_args.kwargs['pass_fds'], (99,))
            self.assertEqual(started.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
            Path(request['result']).unlink()
            failure = {'range_uid': request['range_uid'], 'status': 'failed', 'fatal': True, 'log': 'failure.log'}
            with mock.patch.object(velocity.RangePool, 'execute', return_value=failure):
                with self.assertRaisesRegex(RuntimeError, 'stopped'):
                    velocity.run_requests([request], args, root)
            self.assertEqual(json.loads((root / 'execution.json').read_text())['status'], 'failed')
            with mock.patch.object(velocity.RangePool, 'execute', side_effect=OSError(12, 'Cannot allocate memory')):
                with self.assertRaisesRegex(RuntimeError, 'stopped'):
                    velocity.run_requests([request], args, root)
            failures = json.loads((root / 'failures.json').read_text())
            self.assertEqual(failures[0]['range_uid'], 'uid-a')
            self.assertIn('Cannot allocate memory', failures[0]['error'])

    def test_incomplete_result_never_replaces_existing_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / 'range_summary.csv'
            summary.write_text('last complete summary')
            with mock.patch.object(velocity, 'validate_sources'):
                with self.assertRaisesRegex(ValueError, 'every planned range'):
                    velocity.aggregate([{'range_uid': 'a', 'signature': 'a', 'result': str(root / 'missing.json')}],
                                       None, None, None, root)
            self.assertEqual(summary.read_text(), 'last complete summary')

    def test_unified_selection_prepares_only_vy_and_respects_plot_only_or_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'inputs').mkdir()
            for name in ['arase_phase_master.csv', 'annual_ranges.json', 'provenance.json']:
                (root / 'inputs' / name).touch()
            arguments = ['--year', '2022', '--statistics-dir', str(root), '--product', 'dependence',
                         '--parameter', 'abs_vsys_y', '--workers', '3', '--vsys-y-bin-width-kms', '5', '--overwrite']
            args = unified.build_parser().parse_args(arguments)
            steps = unified.build_steps(args)[-1]
            self.assertEqual([s['name'] for s in steps], ['prepare_vsys_y', 'plot_dependence'])
            self.assertNotIn('--overwrite', steps[0]['command'])
            self.assertEqual(steps[0]['command'][steps[0]['command'].index('--workers') + 1], '3')
            for extra in [['--plot-only'], ['--vsys-y-summary', str(root / 'custom.csv')]]:
                args = unified.build_parser().parse_args([*arguments, *extra])
                self.assertEqual([s['name'] for s in unified.build_steps(args)[-1]], ['plot_dependence'])

    def test_plot_loader_checks_summary_and_signed_median_abs_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'range_summary.csv'
            master = make_frame()
            master['range_uid'] = [f'uid-{i}' for i in range(len(master))]
            master = master.drop(columns=[c for c in master if c.startswith(velocity.PREFIX)])
            summaries = []
            for i, y in enumerate([-30000., -10000., 0., 10000., 20000., 30000.]):
                request = {'range_uid': f'uid-{i}', 'signature': str(i)}
                result = complete(request)
                result.update(velocity.plasma.describe([y, y], velocity.PREFIX))
                summaries.append(result)
            pd.DataFrame(summaries).to_csv(path, index=False)
            velocity.atomic_json(path.with_suffix('.provenance.json'), {'master_sha256': 'master', 'method': velocity.METHOD,
                'summary_sha256': velocity.batch.file_sha256(path)})
            joined = environment.load_derived_summary(master, path, 'master', velocity.METHOD, (velocity.PREFIX,))
            np.testing.assert_allclose(environment.parameter_values(joined, 'vsys_y'), [-30., -10., 0., 10., 20., 30.])
            np.testing.assert_allclose(environment.parameter_values(joined, 'abs_vsys_y'), [30., 10., 0., 10., 20., 30.])
            with self.assertRaisesRegex(ValueError, 'master/method'):
                environment.load_derived_summary(master, path, 'wrong', velocity.METHOD, (velocity.PREFIX,))
            # Exercise the actual plot entrypoint against the synthetic prepared sidecar.
            root = path.parent
            (root / 'inputs').mkdir()
            master_path = root / 'inputs/arase_phase_master.csv'
            master.to_csv(master_path, index=False)
            master_hash = velocity.batch.file_sha256(master_path)
            velocity.atomic_json(root / 'inputs/provenance.json', {'year': 2022,
                'analysis_version': velocity.batch.ANALYSIS_VERSION, 'annual_master_sha256': master_hash})
            velocity.atomic_json(path.with_suffix('.provenance.json'), {'master_sha256': master_hash,
                'method': velocity.METHOD, 'summary_sha256': velocity.batch.file_sha256(path)})
            original = master_path.read_bytes()
            with mock.patch.dict(environment.PLOT_CONFIG, {'dpi': 30}):
                environment.main(['--year', '2022', '--statistics-dir', str(root), '--phase', 'all',
                    '--parameter', 'abs_vsys_y', '--vsys-y-summary', str(path), '--overwrite'])
            directory = root / 'min_blocks_10/abs_vsys_y_statistics'
            table = pd.read_csv(next(directory.glob('*.csv')))
            self.assertTrue(table.x_unit.eq('km/s').all())
            self.assertTrue(table.x_source_transform.eq('abs').all())
            self.assertTrue(table.overall_n.eq(6).all())
            self.assertEqual(master_path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
