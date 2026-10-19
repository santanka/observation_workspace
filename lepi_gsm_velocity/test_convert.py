import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import convert
import core


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name); self.source=self.base/'source'; self.target=self.base/'target'
        self.path=self.source/'1970/01/state/19700101/19700101_000000_19700101_000500.json'
        self.input=self.source/'1970/01/data/19700101/19700101_000000_19700101_000500.npz'
        self.input.parent.mkdir(parents=True)
        manifest={'profile_hash':'native','config':{'frame':'native DSI; spacecraft-relative'}}
        arrays={'time_unix':np.array([4.,12.]),'epoch_tt2000':np.array([1,2],dtype=np.int64),
                'manifest_json':json.dumps(manifest)}
        for var in convert.SPECIES: arrays[var+'__velocity']=np.array([[200.,-10.,2.],[1.,2.,3.]])
        np.savez(self.input,**arrays)
        convert.atomic_json(self.path,{'status':'complete','profile_hash':'native',
            'input_verification':'verified','output_sha256':convert.sha(self.input),
            'start_utc':'1970-01-01T00:00:00+00:00','end_utc_exclusive':'1970-01-01T00:05:00+00:00'})
        self.config={'source_profile_hash':'native','attitude_gap_seconds':16.,'orbit_gap_seconds':120.,
                     'sources':{},'packages':{}}
        self.support=SimpleNamespace(at=np.array([0.,8.,16.]),angles=np.array([[0.,90.]]*3),
            ot=np.array([0.,8.,16.]),positions=np.array([[6371.2,0.,0.]]*3),inputs=[],
            quality=np.array([['I','G','4000','0']]*3),orbit_input_units='re')
        self.output=self.target/'1970/01/data/19700101'/self.input.name

    def tearDown(self): self.temp.cleanup()

    def run_chunk(self):
        def rotate(*args,**kwargs):
            return core.rotation(*args,**kwargs,transform=lambda **kw:kw['data_in'])
        with patch.object(convert,'rotation',side_effect=rotate):
            return convert.convert_chunk(self.path,self.source,self.target,'gsm',self.config,self.support)

    def test_save_three_species_preserve_source_and_resume(self):
        before=convert.sha(self.input)
        self.assertEqual(self.run_chunk(),'complete')
        self.assertEqual(convert.sha(self.input),before)
        with np.load(self.output,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['epoch_tt2000'],[1,2])
            for var in convert.SPECIES:
                np.testing.assert_allclose(saved[var+'__velocity_gsm_observed'],[[200.,-10.,2.],[1.,2.,3.]],atol=1e-12)
            self.assertEqual(saved['attitude_left_flag_vi'][0],'I')
            self.assertEqual(saved['attitude_left_adsg_interp_dt_seconds'][0],4000.)
        self.assertEqual(self.run_chunk(),'skipped_verified')
        self.output.write_bytes(b'corrupt')
        self.assertEqual(self.run_chunk(),'complete')

    def test_partial_nan_output_cannot_masquerade_as_valid(self):
        self.config['attitude_gap_seconds']=4.
        self.assertEqual(self.run_chunk(),'partial')
        with np.load(self.output,allow_pickle=False) as saved:
            self.assertFalse(saved['rotation_valid'].any())
            self.assertTrue(np.isnan(saved['FPDU__velocity_gsm_observed']).all())
            self.assertFalse(saved['FPDU__velocity_valid'].any())
        self.assertEqual(self.run_chunk(),'skipped_partial')

    def make_partial_source(self):
        with np.load(self.input,allow_pickle=False) as saved:
            arrays={key:saved[key] for key in saved.files}
        arrays['moment_valid']=np.array([True,False])
        arrays['moment_invalid_reason']=np.array([0,2],dtype=np.uint8)
        # Finite invalid-row values must be masked even if a source is malformed.
        np.savez(self.input,**arrays)
        state=json.loads(self.path.read_text())
        state.update(status='partial',output_sha256=convert.sha(self.input))
        convert.atomic_json(self.path,state)

    def test_partial_native_source_preserves_mask_and_times(self):
        self.make_partial_source()
        before=convert.sha(self.input)
        self.assertEqual(self.run_chunk(),'partial')
        with np.load(self.output,allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['time_unix'],[4.,12.])
            np.testing.assert_array_equal(saved['moment_valid'],[True,False])
            np.testing.assert_array_equal(saved['moment_invalid_reason'],[0,2])
            for var in convert.SPECIES:
                np.testing.assert_allclose(saved[var+'__velocity_gsm_observed'][0],[200.,-10.,2.],atol=1e-12)
                self.assertTrue(np.isnan(saved[var+'__velocity_gsm_observed'][1]).all())
                np.testing.assert_array_equal(saved[var+'__velocity_valid'],[True,False])
        state=json.loads((self.target/self.path.relative_to(self.source)).read_text())
        self.assertFalse(state['rotation_incomplete'])
        self.assertEqual(state['valid_source_moment_records'],1)
        self.assertEqual(convert.sha(self.input),before)
        self.assertEqual(self.run_chunk(),'skipped_partial')
        a=dt.datetime(1970,1,1,tzinfo=convert.UTC)
        self.assertEqual(len(convert.inventory(self.source,a,a+dt.timedelta(days=1))),1)

    def test_partial_without_mask_or_inconsistent_mask_is_rejected(self):
        state=json.loads(self.path.read_text());state['status']='partial'
        convert.atomic_json(self.path,state)
        with self.assertRaisesRegex(ValueError,'lacks per-record'): self.run_chunk()
        self.make_partial_source()
        state=json.loads(self.path.read_text());state['status']='complete';convert.atomic_json(self.path,state)
        with self.assertRaisesRegex(ValueError,'inconsistent'): self.run_chunk()

    def test_source_hash_corruption_refused(self):
        self.input.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): self.run_chunk()
        self.assertFalse(self.output.exists())

    def test_input_changes_during_publication_refused_and_temp_removed(self):
        with patch.object(convert,'assert_snapshot',side_effect=RuntimeError('input changed')):
            with self.assertRaisesRegex(RuntimeError,'input changed'): self.run_chunk()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.target.rglob('*.tmp')))

    def test_attitude_missing_flags_do_not_shift_quality_columns(self):
        normal='2022-09-29/00:00:04.2897 7.4 16.9 4.0 0 168 78 I G 0.0 -79 59 16.9 4.0 4414.6 0.0'
        omitted='2022-09-29/00:00:12.2897 7.4 16.9 4.0 NaN NaN NaN 359.8 -80 60 16.9 4.0 14133.4 -0.0'
        times,q=convert.attitude_quality_rows('# J2000 axes\n'+normal+'\n'+omitted)
        self.assertEqual(q,[['I','G','4414.6','0.0'],['','','14133.4','-0.0']])
        with self.assertRaises(ValueError): convert.attitude_quality_rows('# J2000\n'+omitted+' garbage')

    def test_leap_year_inventory_ignores_noncomplete_source(self):
        args=argparse.Namespace(month='2020-02',year=None,start=None,end=None)
        a,b=convert.selected_range(args);self.assertEqual((b-a).days,29)
        a=dt.datetime(1970,1,1,tzinfo=convert.UTC);b=a+dt.timedelta(days=1)
        self.assertEqual(len(convert.inventory(self.source,a,b)),1)
        state=json.loads(self.path.read_text());state['status']='running';convert.atomic_json(self.path,state)
        self.assertEqual(convert.inventory(self.source,a,b),[])


if __name__=='__main__': unittest.main()
