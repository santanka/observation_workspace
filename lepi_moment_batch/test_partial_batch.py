"""Local synthetic CDFs through the actual batch scheduler, persistence and resume."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batch import run, profile, fingerprint, sha_file
from acquisition import Acquisition, unit
from test_acquisition import fixture


class PartialBatchTests(unittest.TestCase):
    def test_partial_is_saved_counted_and_resumed(self):
        import numpy as np
        from cdflib import cdfwrite, cdfepoch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=root/'raw';lepi=raw/'lepi/l2/3dflux/2022/09/erg_lepi_l2_3dflux_20220929_v03_00.cdf'
            mgf=raw/'mgf/l2/64hz/2022/09/erg_mgf_l2_64hz_dsi_2022092905_v04.06.cdf'
            lepi.parent.mkdir(parents=True);mgf.parent.mkdir(parents=True)
            fixture(lepi,product='lepi',seconds=(0,4,8,12,16),energy=np.arange(32,0,-1))
            epoch=cdfepoch.compute_tt2000([[2022,9,29,5,5,s,0,0,0] for s in (0,4,8,12,16)])
            b=np.tile([10.,20.,30.],(5,1));b[2]=-1e31
            with cdfwrite.CDF(str(mgf),delete=True) as c:
                for name,typ,dims,data in [('epoch_64hz',33,[],epoch),('mag_64hz_dsi',21,[3],b),('quality_64hz',41,[],np.zeros(5,dtype=np.int8))]:
                    c.write_var({'Variable':name,'Data_Type':typ,'Num_Elements':1,'Rec_Vary':True,'Dim_Sizes':dims},var_data=data)
            start=dt.datetime(2022,9,29,5,5,tzinfo=dt.timezone.utc);end=start+dt.timedelta(minutes=5)
            args=argparse.Namespace(output_root=root/'out',input_root=raw,scratch_root=root/'scratch',
                 mgf_cadence='64hz',max_mgf_gap_seconds=4.,min_free_memory_gib=0.,scratch_reserve_gib=0.,
                 remote_root='https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/',offline=True,
                 compute_workers=1,compute_cpu_ids=sorted(os.sched_getaffinity(0)),pause_seconds=0.,worker_timeout=30.,progress='off')
            identity,config=profile(args)
            proof=[{'unit':unit(product,day)['key'],'status':'verified','fingerprint':fingerprint(path),
                    'sha256':sha_file(path),'validation_identity':'synthetic'}
                   for product,day,path in [('lepi',start.replace(hour=0,minute=0),lepi),('64hz',start.replace(minute=0),mgf)]]
            result=([lepi],[mgf],proof,[],'verified')
            with patch.object(Acquisition,'prefetch',return_value={}),patch.object(Acquisition,'interval',return_value=result),patch.object(Acquisition,'request',side_effect=AssertionError('No network allowed')):
                self.assertEqual(run(args,start,end,identity,config),0)
                output_root=args.output_root/('native_dsi_'+identity[:12])
                report=json.loads((output_root/'last_run.json').read_text())
                self.assertEqual(report['counts'],{'partial':1})
                state_path=next(output_root.glob('2022/09/state/*/*.json'))
                state=json.loads(state_path.read_text());self.assertEqual(state['computed_records_per_species'],4)
                summary=json.loads((output_root/'2022/09/summary.json').read_text())
                self.assertEqual(summary['record_counts'],{'native_records_in_saved_products':5,
                    'computed_records_in_saved_products':4,'invalid_records_in_saved_products':1})
                self.assertEqual(run(args,start,end,identity,config),0)
                report=json.loads((output_root/'last_run.json').read_text())
                self.assertEqual(report['counts'],{'skipped_partial':1})
                self.assertEqual(json.loads(state_path.read_text())['output_sha256'],state['output_sha256'])


if __name__=='__main__':unittest.main()
