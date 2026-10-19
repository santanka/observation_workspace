"""Ordered multi-profile reuse, unusable first candidates and unchanged originals."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import batch


class MultipleReuseTests(unittest.TestCase):
    def make_profile(self, base, second=False):
        sources=dict(batch.REUSABLE_RUNNER_SOURCES)
        if second:
            sources[str(batch.BASE/'batch.py')]=batch.REUSABLE_FAST_RUNNER_SOURCES[str(batch.BASE/'batch.py')]
        config={'conditions':'same','sources':sources}
        identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        root=base/('native_dsi_'+identity[:12])
        batch.atomic_json(root/'profile.json',{'profile_hash':identity,'config':config})
        state=root/'state/day/chunk.json';output=root/'data/day/chunk.npz'
        output.parent.mkdir(parents=True)
        np.savez_compressed(output,epoch_tt2000=np.arange(3,dtype=np.int64),data=np.full(3,2 if second else 1),
                            manifest_json=json.dumps({'profile_hash':identity,'config':config}))
        proof=[{'unit':'lepi_day','status':'verified','fingerprint':{'path':'raw'},'sha256':'same raw','validation_identity':'old'}]
        record={'status':'complete','profile_hash':identity,'input_signature':'old','input_verification':'verified',
                'acquisition_evidence':proof,'output_sha256':batch.sha_file(output)}
        batch.atomic_json(state,record)
        return root,state,output,record

    def test_priority_fallback_and_provenance(self):
        for scenario in ('both_complete','missing_state','failed','excluded','partial','corrupt_output','changed_inputs','neither_complete'):
            with self.subTest(scenario=scenario),tempfile.TemporaryDirectory() as directory:
                base=Path(directory)
                a,sa,oa,record=self.make_profile(base)
                b,sb,ob,other=self.make_profile(base,second=True)
                config={'conditions':'same','sources':{key:'current reviewed source' for key in batch.REUSABLE_RUNNER_SOURCES}}
                candidates=batch.reuse_profiles([a,b,a],config)
                self.assertEqual([x[0] for x in candidates],[a,b])
                self.assertEqual(batch.reuse_profiles(a,config),candidates[:1])
                if scenario=='missing_state':sa.unlink()
                elif scenario in ('failed','excluded','partial','neither_complete'):
                    batch.atomic_json(sa,{**record,'status':'partial' if scenario=='neither_complete' else scenario})
                elif scenario=='corrupt_output':oa.write_bytes(b'broken')
                elif scenario=='changed_inputs':
                    batch.atomic_json(sa,{**record,'acquisition_evidence':[{**record['acquisition_evidence'][0],'sha256':'different raw'}]})
                if scenario=='neither_complete':batch.atomic_json(sb,{**other,'status':'failed_validation'})
                before=[batch.sha_file(p) if p.exists() else None for p in (sa,oa,sb,ob)]
                root=base/'new';state=root/'state/day/chunk.json';output=root/'data/day/chunk.npz'
                common={'input_verification':'verified','acquisition_evidence':record['acquisition_evidence']}
                reused=batch.reuse_first_result(candidates,root,state,output,'new','new signature',common,config)
                self.assertEqual(reused,scenario!='neither_complete')
                if reused:
                    chosen=a if scenario=='both_complete' else b
                    self.assertEqual(json.loads(state.read_text())['reused_from']['profile_hash'],json.loads((chosen/'profile.json').read_text())['profile_hash'])
                    with np.load(output,allow_pickle=False) as data:
                        np.testing.assert_array_equal(data['data'],[1,1,1] if chosen==a else [2,2,2])
                    self.assertTrue(batch.resume_valid(state,output,'new','new signature'))
                else:self.assertFalse(output.exists())
                self.assertEqual([batch.sha_file(p) if p.exists() else None for p in (sa,oa,sb,ob)],before)


if __name__=='__main__':unittest.main()
