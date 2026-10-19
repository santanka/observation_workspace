"""Synthetic CDF only: compare all output arrays at 1/2/4/8 compute workers."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from batch import fingerprint
from compute import ComputePool
from test_acquisition import fixture


def main():
    with tempfile.TemporaryDirectory(prefix='lepi_compute_check_') as directory:
        root=Path(directory)
        lepi,mgf=root/'lepi.cdf',root/'mgf.cdf'
        fixture(lepi,product='lepi')
        fixture(mgf,seconds=(0,4,8,12,16))
        def source(path):
            return {'source':fingerprint(path),'expanded_path':str(path),
                    'time_cache_path':str(path.with_suffix('.epochs.npz'))}
        config={'mgf_cadence':'64hz','max_mgf_gap_seconds':4.,
                'species':{'FPDU':'proton','FHEDU':'heplus','FODU':'oplus'},'energy_range_ev_per_q':[30.,30000.]}
        common={'input_verification':'verified','profile_hash':'synthetic','input_signature':'same-input',
                'start_utc':'2022-09-29T05:05:00+00:00','end_utc_exclusive':'2022-09-29T05:10:00+00:00',
                'acquisition_evidence':[]}
        baseline=None
        cpus=sorted(os.sched_getaffinity(0))
        for workers in (1,2,4,8):
            if workers>len(cpus):continue
            args=argparse.Namespace(compute_workers=workers,compute_cpu_ids=cpus,worker_timeout=60.,pause_seconds=0.)
            results=[];outputs=[];t0=time.monotonic()
            with ComputePool(args,lambda label,state:results.append(state)) as pool:
                for index in range(8):
                    out=root/f'w{workers}_{index}.npz';outputs.append(out)
                    request={**common,'config':config,'output':str(out),'lepi':[source(lepi)],'mgf':[source(mgf)]}
                    pool.submit(str(index),request,common,root/f'w{workers}_{index}.json',
                                root/f'job{workers}_{index}',root/f'w{workers}_{index}.log')
            elapsed=time.monotonic()-t0
            assert len(results)==8 and all(s['status']=='complete' for s in results),results
            assert len({s['cpu_affinity'][0] for s in results})==workers,results
            for path in outputs:
                with np.load(path,allow_pickle=False) as data:
                    arrays={key:data[key] for key in data.files if key!='manifest_json'}
                if baseline is None:baseline=arrays
                assert arrays.keys()==baseline.keys()
                for key in arrays:np.testing.assert_array_equal(arrays[key],baseline[key],err_msg=key)
            print(json.dumps({'workers':workers,'jobs':8,'elapsed_seconds':elapsed,
                  'max_worker_rss_mib':max(s['max_rss_mib'] for s in results),
                  'cpu_ids':sorted({s['cpu_affinity'][0] for s in results}),
                  'identical_arrays_per_chunk':len(baseline)}),flush=True)
        print('Parallel compute passed: all 48 moment series and diagnostics identical; distinct CPU assignment')


if __name__=='__main__':main()
