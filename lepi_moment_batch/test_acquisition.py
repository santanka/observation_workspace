"""Acquisition failure injection, using tiny fixtures and no external network."""
import argparse
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import acquisition
from batch import sha_file

try:
    import numpy as np
    from cdflib import cdfwrite, cdfepoch
    HAS_CDF = True
except ImportError:
    HAS_CDF = False

DATE = dt.datetime(2022, 9, 29, 5, tzinfo=dt.timezone.utc)
NAME = 'erg_mgf_l2_64hz_dsi_2022092905_v04.06.cdf'


def fixture(path, product='64hz', omit=None, seconds=(4, 12), conflict=False, date=(2022, 9, 29, 5, 5)):
    with cdfwrite.CDF(str(path), delete=True) as c:
        epoch = cdfepoch.compute_tt2000([[*date, s, 0, 0, 0] for s in seconds])
        def put(name, dims, data, typ=21, record=True):
            if name != omit:
                if conflict and name == 'mag_64hz_dsi':
                    data[-1, 0] += 1
                c.write_var({'Variable': name, 'Data_Type': typ, 'Num_Elements': 1,
                             'Rec_Vary': record, 'Dim_Sizes': dims},
                            var_attrs={'DEPEND_0': 'Epoch', 'DEPEND_1': name + '_Energy',
                                       'UNITS': '#/s-cm2-sr-keV/q'}
                            if name in ('FPDU', 'FHEDU', 'FODU') else {}, var_data=data)
        if product != 'lepi':
            put('epoch_' + product, [], epoch, 33)
            put('mag_' + product + '_dsi', [3], np.ones((len(seconds), 3), dtype=np.float32))
            put('quality_' + product, [], np.zeros(len(seconds), dtype=np.int8), 41)
        else:
            put('Epoch', [], epoch, 33)
            put('epoch_sp', [], epoch, 33)
            for species in ('FPDU', 'FHEDU', 'FODU'):
                for suffix in ('', '_COUNT_RAW'):
                    put(species + suffix, [32, 8, 16], np.ones((len(seconds), 32, 8, 16), dtype=np.float32))
                put(species + '_Quality', [32, 15, 16], np.zeros((len(seconds), 32, 15, 16), dtype=np.int8), 41)
                put(species + '_Energy', [32], np.arange(1, 33, dtype=np.float32), record=False)


class Response(io.BytesIO):
    status = 200
    def __init__(self, body, url, size=None):
        super().__init__(body)
        self.url = url
        self.headers = {'Content-Length': str(len(body) if size is None else size)}
    def geturl(self):
        return self.url


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(output_root=self.root/'out', input_root=self.root/'raw',
                    remote_root='http://127.0.0.1/erg/', offline=False, refresh_downloads=False,
                    download_attempts=3, retry_delay=0., http_timeout=1., download_timeout=2.,
                    validation_timeout=20., min_free_memory_gib=0., scratch_reserve_gib=0.)
        self.scratch = self.root/'scratch'
        self.scratch.mkdir()
        self.a = acquisition.Acquisition(self.args, self.scratch)
        self.u = acquisition.unit('64hz', DATE)
        self.index_url = self.args.remote_root + self.u['folder'] + '/'
        self.url = self.index_url + NAME
        self.index = ('<html><h1>Index of /erg/' + self.u['folder'] + '</h1><a href="' + NAME +
                      '">cdf</a></html>').encode()
    def tearDown(self):
        self.tmp.cleanup()
    def state(self):
        return next((self.a.root/'state').glob('*/*/*.json'))
    def fail_status(self, failure, expected, count):
        with patch.object(self.a, 'request', side_effect=failure) as request:
            result = self.a.ensure('64hz', DATE)
        self.assertEqual(result['status'], expected)
        self.assertEqual(request.call_count, count)
        self.assertFalse(list(self.args.input_root.rglob('*.cdf')))
        self.assertFalse(list(self.a.root.rglob('*.part')))
    def test_authorization_not_absence_and_no_retry(self):
        for code in (401, 403):
            self.a.results.clear()
            self.a.catalogs.clear()
            self.fail_status(urllib.error.HTTPError(self.index_url, code, 'denied', {}, None),
                             'failed_authorization', 1)
    def test_timeout_and_503_retry_bounded(self):
        for error in (TimeoutError('read timed out'),
                      urllib.error.HTTPError(self.index_url, 503, 'busy', {}, None)):
            self.a.results.clear()
            self.a.catalogs.clear()
            self.fail_status(error, 'failed_download', 3)
    def test_directory_failure_does_not_poison_next_hour(self):
        with patch.object(self.a, 'request', side_effect=TimeoutError('read timed out')) as request:
            self.assertEqual(self.a.ensure('64hz', DATE)['status'], 'failed_download')
            self.assertEqual(request.call_count, 3)
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET': Response(self.index, url)) as request:
            self.assertEqual(self.a.ensure('64hz', DATE + dt.timedelta(hours=1))['status'], 'remote_absent')
            self.assertEqual(request.call_count, 1)

    def test_persistent_network_failure_stops_instead_of_skipping_year(self):
        self.args.max_network_failures = 3
        with patch.object(self.a, 'request', side_effect=TimeoutError('read timed out')):
            for hour in range(2):
                self.assertEqual(self.a.ensure('64hz', DATE + dt.timedelta(hours=hour))['status'], 'failed_download')
            with self.assertRaisesRegex(acquisition.AcquisitionError, 'Stopping after 3'):
                self.a.ensure('64hz', DATE + dt.timedelta(hours=2))
        self.assertEqual(len(list((self.a.root/'state').glob('*/*/*.json'))), 3)

    def test_default_defers_failure_and_retry_removes_only_transient_units(self):
        with patch.object(self.a, 'request', side_effect=TimeoutError('outage')):
            for hour in range(4):
                self.assertEqual(self.a.ensure('64hz', DATE + dt.timedelta(hours=hour))['status'], 'failed_download')
        self.a.results['policy'] = {'status': 'failed_validation', 'retryable': False}
        self.a.results['absent'] = {'status': 'remote_absent'}
        self.assertEqual(len(self.a.retry_failed()), 4)
        self.assertEqual(set(self.a.results), {'policy', 'absent'})

    def test_prefetch_deduplicates_units_before_processing(self):
        day = DATE.replace(hour=0)
        with patch.object(self.a, 'ensure', side_effect=lambda product, date:
                          {'unit': acquisition.unit(product, date), 'status': 'remote_absent'}) as ensure:
            outcomes = self.a.prefetch([(day, DATE, DATE + dt.timedelta(minutes=10)),
                                       (day, DATE, DATE + dt.timedelta(minutes=5))], '64hz', 'off')
        self.assertEqual(ensure.call_count, len(outcomes))
        self.assertEqual(set(outcomes), {'lepi_20220929', '64hz_2022092904', '64hz_2022092905'})

    def test_login_html_and_incomplete_index_not_absence(self):
        for body in (b'<html>login</html>', self.index[:-7]):
            self.a.results.clear()
            self.a.catalogs.clear()
            self.fail_status(lambda *a: Response(body, self.index_url), 'failed_download', 3)
    def test_confirmed_absence_is_refreshed_next_invocation(self):
        empty = ('<html><h1>Index of /erg/' + self.u['folder'] + '</h1></html>').encode()
        with patch.object(self.a, 'request', return_value=Response(empty, self.index_url)):
            self.assertEqual(self.a.ensure('64hz', DATE)['status'], 'remote_absent')
        other = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(other, 'request', side_effect=TimeoutError('offline')) as request:
            self.assertEqual(other.ensure('64hz', DATE)['status'], 'failed_download')
            self.assertEqual(request.call_count, 3)
    def test_directory404_is_absence_but_listed_file404_is_failure(self):
        self.fail_status(urllib.error.HTTPError(self.index_url, 404, 'absent', {}, None), 'remote_absent', 1)
        self.a.results.clear()
        self.a.catalogs.clear()
        def transport(url, method='GET'):
            if url == self.index_url:
                return Response(self.index, url)
            raise urllib.error.HTTPError(url, 404, 'gone', {}, None)
        with patch.object(self.a, 'request', side_effect=transport):
            result = self.a.ensure('64hz', DATE)
        self.assertEqual(result['status'], 'failed_download')
    def test_http_wall_timeout_and_unknown_boundary_block(self):
        import time
        class Slow(Response):
            def read(self, size=-1):
                time.sleep(0.1)
                return super().read(size)
        self.args.http_timeout = 0.01
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET': Slow(self.index, url)):
            self.assertEqual(self.a.ensure('64hz', DATE)['status'], 'failed_download')
        day = DATE.replace(hour=0)
        def ensure(product, date):
            u = acquisition.unit(product, date)
            if date < day:
                return {'status': 'failed_download', 'unit': u, 'reason': 'timeout'}
            return {'status': 'remote_absent', 'unit': u, 'evidence': {}}
        with patch.object(self.a, 'ensure', side_effect=ensure):
            _, _, proof, failures, _ = self.a.interval(day, day, day + dt.timedelta(minutes=5), '64hz')
            self.assertEqual(failures[0]['unit']['key'], 'lepi_20220928')

    def test_short_body_never_published(self):
        def transport(url, method='GET'):
            return Response(self.index, url) if url == self.index_url else Response(b'partial', url, 1000)
        with patch.object(self.a, 'request', side_effect=transport):
            result = self.a.ensure('64hz', DATE)
        self.assertEqual(result['status'], 'failed_download')
        self.assertFalse(list(self.a.root.rglob('*.part')))
        self.assertFalse(list(self.args.input_root.rglob('*.cdf')))
    def test_one_failed_file_does_not_poison_month_catalog(self):
        def transport(url, method='GET'):
            if url == self.index_url:
                return Response(self.index, url)
            raise TimeoutError('file read timed out')
        with patch.object(self.a, 'request', side_effect=transport):
            self.assertEqual(self.a.ensure('64hz', DATE)['status'], 'failed_download')
            self.assertEqual(self.a.ensure('64hz', DATE + dt.timedelta(hours=1))['status'], 'remote_absent')

    def test_offline_never_trusts_unverified_local_file(self):
        self.args.offline = True
        local = self.args.input_root/self.u['folder']/NAME
        local.parent.mkdir(parents=True)
        local.write_bytes(b'unverified')
        with patch.object(self.a, 'request') as request:
            self.assertEqual(self.a.ensure('64hz', DATE)['status'], 'failed_download')
            request.assert_not_called()
        self.assertEqual(local.read_bytes(), b'unverified')

    @unittest.skipUnless(HAS_CDF, 'Run with .venv_pyspedas/bin/python for real CDF validator')
    def test_corrupt_local_retry_then_verified_resume_and_cache_corruption(self):
        fixture_path = self.root/'fixture.cdf'
        fixture(fixture_path)
        body = fixture_path.read_bytes()
        local = self.args.input_root/self.u['folder']/NAME
        local.parent.mkdir(parents=True)
        local.write_bytes(b'x' * len(body))
        before = sha_file(local)
        methods = []
        def transport(url, method='GET'):
            methods.append(method)
            return Response(self.index, url) if url == self.index_url else Response(body if method == 'GET' else b'', url, len(body))
        with patch.object(self.a, 'request', side_effect=transport):
            result = self.a.ensure('64hz', DATE)
        self.assertEqual(result['status'], 'verified', result)
        self.assertEqual(result['origin'], 'http_download')
        self.assertEqual(result['attempts'][0]['status'], 'failed_validation')
        self.assertIn('HEAD', methods)
        self.assertEqual(sha_file(local), result['sha256'])
        history = list((self.a.root/'history').rglob('*.cdf'))
        self.assertTrue(any(sha_file(path) == before for path in history))
        other = acquisition.Acquisition(self.args, self.scratch)
        self.args.offline = True
        with patch.object(other, 'request') as request:
            self.assertEqual(other.ensure('64hz', DATE)['sha256'], result['sha256'])
            request.assert_not_called()
        # Same size + restored mtime still must fail SHA256 check.
        target = Path(result['path'])
        original = target.stat()
        target.write_bytes(b'y' * len(body))
        import os
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        other = acquisition.Acquisition(self.args, self.scratch)
        self.assertEqual(other.ensure('64hz', DATE)['status'], 'failed_download')
        self.args.offline = False
        other = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(other, 'request', side_effect=transport):
            repaired = other.ensure('64hz', DATE)
        self.assertEqual(repaired['status'], 'verified', repaired)
        self.assertEqual(sha_file(target), result['sha256'])

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_legacy_verified_cache_migrates_offline_to_spedas_layout(self):
        fixture_path = self.root/'fixture.cdf'
        fixture(fixture_path)
        body = fixture_path.read_bytes()
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET':
                          Response(self.index if url == self.index_url else body, url)):
            result = self.a.ensure('64hz', DATE)
        shared = Path(result['path'])
        legacy = self.a.root/'erg'/self.u['folder']/shared.name
        legacy.parent.mkdir(parents=True)
        shared.rename(legacy)
        result.update(path=str(legacy), fingerprint=acquisition.fingerprint(legacy))
        receipt = self.a.root/'receipts'/DATE.strftime('%Y/%m')/(self.u['key']+'.json')
        acquisition.atomic_json(self.state(), result)
        acquisition.atomic_json(receipt, result)
        self.args.offline = True
        other = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(other, 'request') as request:
            migrated = other.ensure('64hz', DATE)
            request.assert_not_called()
        self.assertEqual(Path(migrated['path']), shared)
        self.assertEqual(shared.read_bytes(), body)
        self.assertEqual(legacy.read_bytes(), body)
        self.assertEqual(json.loads(receipt.read_text())['path'], str(shared))

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_refresh_identical_bytes_preserves_input_identity(self):
        path = self.root/'fixture.cdf'
        fixture(path)
        body = path.read_bytes()
        def transport(url, method='GET'):
            return Response(self.index if url == self.index_url else body, url)
        with patch.object(self.a, 'request', side_effect=transport):
            original = self.a.ensure('64hz', DATE)
        self.args.refresh_downloads = True
        other = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(other, 'request', side_effect=transport):
            refreshed = other.ensure('64hz', DATE)
        self.assertEqual(refreshed['status'], 'verified')
        self.assertEqual(refreshed['fingerprint'], original['fingerprint'])
        self.assertEqual(refreshed['sha256'], original['sha256'])

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_valid_length_cdf_missing_variable_never_published(self):
        fixture_path = self.root/'fixture.cdf'
        fixture(fixture_path, omit='quality_64hz')
        body = fixture_path.read_bytes()
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET':
                          Response(self.index if url == self.index_url else body, url)):
            result = self.a.ensure('64hz', DATE)
        self.assertEqual(result['status'], 'failed_validation', result)
        self.assertFalse(list(self.args.input_root.rglob('*.cdf')))

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_identical_duplicate_is_verified_and_conflict_is_quarantined_once(self):
        path = self.root/'fixture.cdf'
        fixture(path, seconds=(4, 4))
        _, report = self.a.verify(path, self.u)
        self.assertEqual(report['identical_duplicate_pairs'], 1)
        fixture(path, seconds=(4, 4), conflict=True)
        body = path.read_bytes()
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET':
                          Response(self.index if url == self.index_url else body, url)) as request:
            result = self.a.ensure('64hz', DATE)
        self.assertEqual(result['status'], 'failed_validation')
        self.assertFalse(result['retryable'])
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(request.call_count, 2)  # index + file; no identical re-downloads
        self.assertEqual(Path(result['rejected_path']).read_bytes(), body)
        self.assertFalse(list(self.args.input_root.rglob('*.cdf')))

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_epoch_reversal_is_policy_failure_without_retry(self):
        path = self.root/'fixture.cdf'
        fixture(path, seconds=(12, 4))
        with self.assertRaises(acquisition.AcquisitionError) as caught:
            self.a.verify(path, self.u)
        self.assertFalse(caught.exception.retryable)

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_new_validator_revalidates_existing_cache_offline(self):
        path = self.root/'fixture.cdf'
        fixture(path)
        body = path.read_bytes()
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET':
                          Response(self.index if url == self.index_url else body, url)):
            result = self.a.ensure('64hz', DATE)
        state = self.state()
        stored = json.loads(state.read_text());stored['validation_identity'] = 'older_validator'
        state.write_text(json.dumps(stored))
        self.args.offline = True
        other = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(other, 'request') as request:
            updated = other.ensure('64hz', DATE)
            self.assertEqual(updated['status'], 'verified')
            request.assert_not_called()
        self.assertIn('revalidated_utc', updated)
        self.assertEqual(updated['sha256'], result['sha256'])

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_lepi_boundary_integration_context_keeps_native_times(self):
        from validate_input import validate, DataPolicyError
        path = self.root/'fixture.cdf'
        fixture(path, product='lepi', seconds=(23, 55), date=(2022, 9, 28, 23, 59))
        report = validate(path, 'lepi', '2022-09-29')
        self.assertEqual(report['boundary_tolerance_seconds'], 64.)
        self.assertEqual(report['first_epoch_utc'], '2022-09-28T23:59:23.000000000')
        with self.assertRaises(DataPolicyError):
            validate(path, 'lepi', '2022-09-29', 32.)

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_failed_refresh_preserves_verified_receipt_for_offline_resume(self):
        path = self.root/'fixture.cdf'
        fixture(path)
        body = path.read_bytes()
        with patch.object(self.a, 'request', side_effect=lambda url, method='GET':
                          Response(self.index if url == self.index_url else body, url)):
            original = self.a.ensure('64hz', DATE)
        self.args.refresh_downloads = True
        refreshed = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(refreshed, 'request', side_effect=TimeoutError('temporary outage')):
            self.assertEqual(refreshed.ensure('64hz', DATE)['status'], 'failed_download')
        self.args.offline = True
        self.args.refresh_downloads = False
        resumed = acquisition.Acquisition(self.args, self.scratch)
        with patch.object(resumed, 'request') as request:
            result = resumed.ensure('64hz', DATE)
            self.assertEqual(result['status'], 'verified')
            self.assertEqual(result['sha256'], original['sha256'])
            request.assert_not_called()

    @unittest.skipUnless(HAS_CDF, 'CDF library required')
    def test_lepi_full_schema_and_partial_day_allowed(self):
        path = self.root/'particles.cdf'
        fixture(path, product='lepi')
        staged, report = self.a.verify(path, acquisition.unit('lepi', DATE.replace(hour=0)))
        self.assertEqual(report['records'], 2)
        self.assertTrue(report['full_required_payload_read'])
        with path.open('ab') as f:
            f.write(b'invalid trailing bytes')
        with self.assertRaises(acquisition.AcquisitionError):
            self.a.verify(path, acquisition.unit('lepi', DATE.replace(hour=0)))


if __name__ == '__main__':
    unittest.main()
