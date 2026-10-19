"""Classification, bounded compute scheduling, timeouts and interruption."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import batch
import compute


class ComputeTests(unittest.TestCase):
    def args(self, workers=2, timeout=3.):
        cpus = sorted(os.sched_getaffinity(0))
        return argparse.Namespace(compute_workers=min(workers, len(cpus)), compute_cpu_ids=cpus,
                                  worker_timeout=timeout, pause_seconds=0.)

    def fake_worker(self, root):
        (root/'worker.py').write_text('''
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--result');a=p.parse_args()
r=json.loads(Path(a.request).read_text())
Path(r['marker']).write_text(str(os.getpid()))
os.sched_setaffinity(0,{int(os.environ['LEPI_COMPUTE_CPU'])})
time.sleep(r.get('sleep',.05))
Path(a.result).write_text(json.dumps({'status':'complete','cpu_affinity':sorted(os.sched_getaffinity(0))}))
''')

    def test_pool_is_bounded_and_uses_distinct_cpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fake_worker(root);finished=[];args=self.args()
            with patch.object(compute, 'BASE', root):
                with compute.ComputePool(args, lambda label,state: finished.append(state)) as pool:
                    for i in range(6):
                        pool.submit(str(i), {'marker':str(root/f'marker{i}')}, {}, root/f'state{i}.json', root/f'job{i}', root/f'log{i}')
                        self.assertLessEqual(len(pool.pending),args.compute_workers)
            self.assertEqual(len(finished),6)
            self.assertEqual({s['status'] for s in finished},{'complete'})
            self.assertEqual(len({s['cpu_affinity'][0] for s in finished}),args.compute_workers)

    def test_timeout_is_failed_and_process_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fake_worker(root);finished=[]
            with patch.object(compute, 'BASE', root):
                with compute.ComputePool(self.args(1,.15), lambda label,state: finished.append(state)) as pool:
                    pool.submit('one',{'marker':str(root/'pid'),'sleep':30}, {},root/'state.json',root/'job',root/'log')
            self.assertEqual(finished[0]['status'],'timeout')
            self.assertEqual(finished[0]['failure_class'],'compute_timeout')
            self.assertTrue(finished[0]['retryable'])
            with self.assertRaises(ProcessLookupError):os.kill(int((root/'pid').read_text()),0)

    def test_interrupt_records_state_before_pool_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fake_worker(root);finished=[]
            with patch.object(compute, 'BASE', root):
                with self.assertRaises(KeyboardInterrupt):
                    with compute.ComputePool(self.args(1),lambda label,state:finished.append(state)) as pool:
                        pool.submit('one',{'marker':str(root/'pid'),'sleep':30},{},root/'state.json',root/'job',root/'log')
                        deadline=time.monotonic()+5
                        while not (root/'pid').exists():
                            if time.monotonic()>deadline:raise AssertionError('worker did not start')
                            time.sleep(.01)
                        raise KeyboardInterrupt('test')
            self.assertEqual(finished[0]['status'],'interrupted')
            self.assertEqual(json.loads((root/'state.json').read_text())['status'],'interrupted')
            with self.assertRaises(ProcessLookupError):os.kill(int((root/'pid').read_text()),0)

    def test_resume_exclusions_requires_evidence_and_matching_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=root/'state.json';out=root/'data.npz'
            data={'status':'excluded','profile_hash':'p','input_signature':'s','retryable':False,
                  'classification_version':1,'exclusion_basis':'verified_cdf_timestamps'}
            batch.atomic_json(state,data)
            self.assertTrue(batch.resume_valid(state,out,'p','s'))
            self.assertFalse(batch.resume_valid(state,out,'p','changed'))
            batch.atomic_json(state,{**data,'exclusion_basis':'unknown'})
            self.assertFalse(batch.resume_valid(state,out,'p','s'))
            batch.atomic_json(state,{**data,'status':'no_data'})
            self.assertFalse(batch.resume_valid(state,out,'p','s'))

    def test_magnetic_exclusion_and_validation_are_separate(self):
        script = """
import worker
r={'mgf':[],'acquisition_evidence':[{'unit':'64hz_day','status':'remote_absent'}]}
exc=worker.MissingMagneticData('no samples','empty_mgf_data')
a=worker.magnetic_failure_result(r,exc)
assert a['status']=='excluded' and a['failure_class']=='required_input_missing_remote'
b=worker.magnetic_failure_result({'mgf':[],'acquisition_evidence':[]},exc)
assert b['status']=='failed_validation' and b['retryable']
for name in ('invalid_mgf_schema','invalid_mgf_field'):
    c=worker.magnetic_failure_result(r,worker.MissingMagneticData('bad field',name))
    assert c['status']=='failed_validation' and c['failure_class']==name
r['mgf']=[{'verified':'CDF'}]
c=worker.magnetic_failure_result(r,worker.MissingMagneticData('gap','insufficient_mgf_coverage'))
assert c['status']=='excluded' and c['exclusion_basis']=='verified_mgf_coverage'
"""
        import subprocess, sys
        subprocess.run([sys.executable,'-c',script],cwd=batch.BASE,check=True,timeout=15)

    def test_download_failures_never_become_exclusions(self):
        cases=[('failed_download','read timed out','catalog','catalog_timeout'),
               ('failed_download','HTTP wall timeout','file','download_timeout'),
               ('failed_authorization','403','catalog','authorization_error'),
               ('failed_validation','bad CDF','file','input_validation_error')]
        for status,reason,phase,expected in cases:
            self.assertEqual(batch.acquisition_failure_class([{'status':status,'attempts':[{'reason':reason,'phase':phase}]}]),expected)


if __name__=='__main__':unittest.main()
