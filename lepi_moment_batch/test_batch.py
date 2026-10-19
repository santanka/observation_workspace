"""Focused orchestration tests; science arrays are checked with real-data smoke runs."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
import batch


class BatchTests(unittest.TestCase):
    def args(self, **kwargs):
        defaults = {'month': None, 'year': None, 'start': None, 'end': None}
        return argparse.Namespace(**(defaults | kwargs))

    def test_leap_month_and_year(self):
        a, b = batch.requested_range(self.args(month='2020-02'))
        self.assertEqual(len(list(batch.days(a, b))), 29)
        self.assertEqual(sum(len(list(batch.chunks(lo, hi))) for _, lo, hi in batch.days(a, b)), 29*288)
        a, b = batch.requested_range(self.args(year=2020))
        self.assertEqual(len(list(batch.days(a, b))), 366)

    def test_year_end_and_offsets(self):
        a, b = batch.requested_range(self.args(month='2022-12'))
        self.assertEqual(b.isoformat(), '2023-01-01T00:00:00+00:00')
        a, b = batch.requested_range(self.args(start='2022-09-29T14:05:00+09:00',
                                             end='2022-09-29T14:15:00+09:00'))
        self.assertEqual(a.hour, 5)
        self.assertEqual(len(list(batch.chunks(a, b))), 2)

    def test_reject_unaligned_or_ambiguous_interval(self):
        with self.assertRaises(ValueError):
            batch.requested_range(self.args(start='2022-09-29T05:06', end='2022-09-29T05:15'))
        with self.assertRaises(ValueError):
            batch.requested_range(self.args(month='2022-09', end='2022-10-01'))
        with self.assertRaises(ValueError):
            batch.requested_range(self.args(year=0))

    def test_latest_numeric_version(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            for name in ['erg_v09_99.cdf', 'erg_v10_00.cdf', 'erg_v03_00.cdf']:
                (p/name).touch()
            self.assertEqual(batch.latest(p, '*.cdf').name, 'erg_v10_00.cdf')

    def test_reject_source_edit_during_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.py'
            path.write_text('version one')
            config = {'sources': {str(path): batch.sha_file(path)}}
            batch.check_sources(config)
            path.write_text('version two')
            with self.assertRaises(RuntimeError):
                batch.check_sources(config)

    def test_resume_checks_inputs_and_file_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            out, state = p/'data.npz', p/'state.json'
            out.write_bytes(b'original data')
            batch.atomic_json(state, {'status': 'complete', 'profile_hash': 'profile',
                                     'input_signature': 'inputs', 'input_verification': 'verified', 'output_sha256': batch.sha_file(out)})
            self.assertTrue(batch.resume_valid(state, out, 'profile', 'inputs'))
            self.assertFalse(batch.resume_valid(state, out, 'profile', 'new inputs'))
            out.write_bytes(b'corrupted')
            self.assertFalse(batch.resume_valid(state, out, 'profile', 'inputs'))
            out.unlink()
            batch.atomic_json(state, {'status': 'no_data', 'profile_hash': 'profile',
                                     'input_signature': 'inputs'})
            self.assertFalse(batch.resume_valid(state, out, 'profile', 'inputs'))
            batch.atomic_json(state, {'status': 'no_data', 'profile_hash': 'profile',
                                     'input_signature': 'inputs', 'input_verification': 'verified'})
            self.assertFalse(batch.resume_valid(state, out, 'profile', 'inputs'))
            self.assertFalse(batch.resume_valid(state, out, 'profile', 'changed inputs'))

    def test_partial_resume_checks_hash_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state.json';out=root/'data.npz'
            out.write_bytes(b'partial product')
            batch.atomic_json(state,{'status':'partial','profile_hash':'p','input_signature':'s',
                                    'input_verification':'verified','output_sha256':batch.sha_file(out)})
            self.assertTrue(batch.resume_valid(state,out,'p','s'))
            self.assertFalse(batch.resume_valid(state,out,'p','new'))
            out.write_bytes(b'changed')
            self.assertFalse(batch.resume_valid(state,out,'p','s'))

    def test_abort_report_retains_current_run_counts(self):
        from collections import Counter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start, end = batch.parse_utc('2022-09-01'), batch.parse_utc('2022-10-01')
            counts = Counter(failed_download=2)
            with self.assertRaises(RuntimeError):
                with batch.track_run(root, start, end, 'profile', counts):
                    raise RuntimeError('network circuit open')
            report = json.loads((root/'last_run.json').read_text())
            self.assertEqual(report['status'], 'stopped')
            self.assertEqual(report['counts'], {'failed_download': 2})
            self.assertEqual(report['reason'], 'network circuit open')

    def test_boundary_granules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = batch.parse_utc('2022-09-30')
            for date in [day-dt.timedelta(days=1), day, day+dt.timedelta(days=1)]:
                folder = root/'lepi/l2/3dflux'/date.strftime('%Y/%m')
                folder.mkdir(parents=True, exist_ok=True)
                (folder/('erg_lepi_l2_3dflux_'+date.strftime('%Y%m%d')+'_v03_00.cdf')).touch()
            first = batch.particle_sources(root, day, day, day+dt.timedelta(minutes=5))
            last = batch.particle_sources(root, day, day+dt.timedelta(hours=23, minutes=55),
                                          day+dt.timedelta(days=1))
            self.assertEqual(len(first), 2)
            self.assertEqual(len(last), 2)
            self.assertIn('20221001', last[-1].name)
            for hour in [day.replace(hour=23), day+dt.timedelta(days=1)]:
                folder = root/'mgf/l2/64hz'/hour.strftime('%Y/%m')
                folder.mkdir(parents=True, exist_ok=True)
                (folder/('erg_mgf_l2_64hz_dsi_'+hour.strftime('%Y%m%d%H')+'_v04.06.cdf')).touch()
            mag = batch.magnetic_sources(root, day.replace(hour=23, minute=55),
                                          day+dt.timedelta(days=1), '64hz')
            self.assertEqual(len(mag), 2)


if __name__ == '__main__':
    unittest.main()
