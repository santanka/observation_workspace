"""Frozen sparse recovery, offline provenance, selected-only scheduling and resume."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch
from acquisition import Acquisition, unit
from recovery_plan import CONDITIONS, SCHEMA, digest, load, ranges
from test_acquisition import fixture


class RecoveryPlanTests(unittest.TestCase):
    def setup_plan(self, root):
        start=dt.datetime(2022,9,29,5,5,tzinfo=dt.timezone.utc)
        args=argparse.Namespace(output_root=root/'out',input_root=root/'raw',scratch_root=root/'scratch',
             recovery_plan=root/'plan.json',offline=True,refresh_downloads=False,
             mgf_cadence='64hz',max_mgf_gap_seconds=4.,lepi_boundary_tolerance_seconds=64.,
             min_free_memory_gib=0.,scratch_reserve_gib=0.,validation_timeout=20.,
             remote_root='https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/',
             compute_workers=1,compute_cpu_ids=sorted(os.sched_getaffinity(0)),pause_seconds=0.,
             worker_timeout=30.,progress='off',download_workers=8)
        identity, config=batch.profile(args)
        plan={'schema':SCHEMA,'conditions':{k:config[k] for k in CONDITIONS},'ranges':[], 'inputs':{}}
        for offset in (0,10):
            lo=start+dt.timedelta(minutes=offset)
            plan['ranges'].append({'start_utc':lo.isoformat(),'end_utc_exclusive':(lo+dt.timedelta(minutes=5)).isoformat()})
        for product,date in [('lepi',start.replace(hour=0,minute=0)),('64hz',start.replace(minute=0))]:
            u=unit(product,date)
            path=root/(u['prefix']+'_v03_00.cdf')
            fixture(path,product=product,seconds=(0,4,8,12,16),energy=list(range(32,0,-1)))
            plan['inputs'][u['key']]={'status':'local_candidate','unit':u,'path':str(path),
                                     'fingerprint':batch.fingerprint(path),'sha256':batch.sha_file(path)}
        plan['plan_sha256']=digest(plan)
        batch.atomic_json(args.recovery_plan,plan)
        return args,identity,config,plan,start

    def test_only_selected_chunks_and_offline_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,identity,config,plan,start=self.setup_plan(root)
            stop=start+dt.timedelta(minutes=15)
            with patch.object(Acquisition,'request',side_effect=AssertionError('Network forbidden')),\
                 patch.object(Acquisition,'catalog',side_effect=AssertionError('Index forbidden')):
                self.assertEqual(batch.run(args,start,stop,identity,config),0)
                output=args.output_root/('native_dsi_'+identity[:12])
                states=sorted(output.glob('2022/09/state/*/*.json'))
                self.assertEqual(len(states),2)
                receipts=list((args.output_root/'verified_inputs/recovery_validations').glob('*/*.json'))
                self.assertEqual(len(receipts),2)
                self.assertTrue(all(json.loads(p.read_text())['validation']['full_required_payload_read'] for p in receipts))
                self.assertEqual([json.loads(p.read_text())['status'] for p in states],['complete','excluded'])
                self.assertEqual([p.name.split('_')[1] for p in states],['050500','051500'])
                self.assertEqual(json.loads((output/'last_run.json').read_text())['counts'],{'complete':1,'excluded':1})
                self.assertEqual(json.loads((output/'last_run.json').read_text())['selected_chunks'],2)
                self.assertEqual(json.loads((output/'last_run.json').read_text())['recovery_plan_sha256'],plan['plan_sha256'])
                self.assertEqual(json.loads((output/'recovery_plan.json').read_text()),plan)
                self.assertEqual(batch.run(args,start,stop,identity,config),0)
                self.assertEqual(json.loads((output/'last_run.json').read_text())['counts'],{'skipped_verified':1,'skipped_excluded':1})

    def test_changed_missing_unplanned_and_historical_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,_,_,plan,start=self.setup_plan(root)
            args.scratch_root.mkdir()
            a=Acquisition(args,args.scratch_root)
            u=unit('64hz',start.replace(minute=0));entry=a.recovery_inputs[u['key']]
            with patch.object(a,'request',side_effect=AssertionError('Network forbidden')):
                entry['sha256']='wrong'
                self.assertEqual(a.ensure('64hz',start.replace(minute=0))['status'],'failed_validation')
                a.results.clear();Path(entry['path']).unlink()
                self.assertEqual(a.ensure('64hz',start.replace(minute=0))['status'],'failed_download')
                other=start.replace(hour=6,minute=0)
                self.assertEqual(a.ensure('64hz',other)['status'],'failed_validation')
                absent=unit('64hz',other)
                a.results.clear();a.recovery_inputs[absent['key']]={'unit':absent,'status':'remote_absent',
                      'evidence':{'url':'https://example.invalid/index','http_status':200,'index_sha256':'recorded index hash'}}
                self.assertEqual(a.ensure('64hz',other)['status'],'remote_absent')

    def test_plan_identity_grid_and_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args,identity,config,plan,start=self.setup_plan(root)
            self.assertEqual(len(ranges(load(args.recovery_plan))),2)
            plan['ranges'].append(plan['ranges'][0])
            batch.atomic_json(args.recovery_plan,plan)
            with self.assertRaisesRegex(ValueError,'identity'):load(args.recovery_plan)
            plan['plan_sha256']=digest(plan);batch.atomic_json(args.recovery_plan,plan)
            with self.assertRaisesRegex(ValueError,'duplicate'):load(args.recovery_plan)
            plan['ranges'].pop();plan['plan_sha256']=digest(plan);batch.atomic_json(args.recovery_plan,plan)
            with self.assertRaisesRegex(ValueError,'conditions'):
                batch.run(args,start,start+dt.timedelta(minutes=15),identity,{**config,'max_mgf_gap_seconds':1.})


if __name__=='__main__':unittest.main()
