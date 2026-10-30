"""Local plan preparation from receipts, rejected CDFs and absence evidence."""
import argparse
import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from acquisition import Acquisition,unit
from batch import atomic_json,fingerprint,sha_file
from prepare_local_plan import prepare,local_entry
from recovery_plan import digest

UTC=dt.timezone.utc


class LocalPlanTests(unittest.TestCase):
    def test_prepare_and_changed_receipt_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            start=dt.datetime(2022,9,29,5,5,tzinfo=UTC);end=start+dt.timedelta(minutes=5)
            args=argparse.Namespace(output_root=root,mgf_cadence='64hz',max_mgf_gap_seconds=1.,
                                    lepi_boundary_tolerance_seconds=64.,workers=2)
            for product,date in [('lepi',start.replace(hour=0,minute=0)),('64hz',start.replace(minute=0))]:
                u=unit(product,date);path=root/(u['key']+'.cdf');path.write_bytes(b'local only')
                row={'status':'verified','unit':u,'path':str(path),'fingerprint':fingerprint(path),'sha256':sha_file(path)}
                atomic_json(root/'verified_inputs/receipts/2022/09'/(u['key']+'.json'),row)
            with patch.object(Acquisition,'request',side_effect=AssertionError('No network')):
                plan=prepare(args,start,end)
            self.assertEqual(len(plan['ranges']),1)
            self.assertEqual(len(plan['inputs']),2)
            self.assertEqual(plan['plan_sha256'],digest(plan))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'unresolved'):prepare(args,start,end)

    def test_rejected_and_absent_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);date=dt.datetime(2022,9,29,5,tzinfo=UTC);u=unit('64hz',date)
            path=root/'rejected.cdf';path.write_bytes(b'old rejection')
            state=root/'verified_inputs/state/2022/09'/(u['key']+'.json')
            atomic_json(state,{'status':'failed_validation','unit':u,'transport_length_verified':True,
                              'rejected_path':str(path),'rejected_sha256':sha_file(path)})
            self.assertEqual(local_entry(root,u)['origin'],'previously_rejected_cdf')
            atomic_json(state,{'status':'remote_absent','unit':u,'evidence':{
                'url':'https://example.invalid/index','http_status':200,'index_sha256':'snapshot','names':['not copied']}})
            entry=local_entry(root,u)
            self.assertEqual(entry['status'],'remote_absent');self.assertNotIn('names',entry['evidence'])
            atomic_json(state,{'status':'failed_download','unit':u,'reason':'timeout'})
            with self.assertRaisesRegex(ValueError,'proven-absence'):local_entry(root,u)


if __name__=='__main__':unittest.main()
