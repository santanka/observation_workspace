"""Synthetic native-grid checks: preserve times, per-record reasons and valid moments."""
import json
from pathlib import Path
import tempfile
import worker
from batch import fingerprint, MOMENT_INVALID_BITS
from test_acquisition import fixture
from check_worker_edges import write_field

np=worker.np


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);lepi=root/'lepi.cdf'
        fixture(lepi,product='lepi',seconds=(0,4,8,12,16),energy=np.arange(32,0,-1))
        source={'source':fingerprint(lepi),'expanded_path':str(lepi)}
        request={'input_verification':'verified','profile_hash':'synthetic','input_signature':'same',
                 'acquisition_evidence':[], 'start_utc':'2022-09-29T05:05:00+00:00',
                 'end_utc_exclusive':'2022-09-29T05:10:00+00:00','lepi':[source],
                 'config':{'mgf_cadence':'64hz','max_mgf_gap_seconds':4.,
                           'species':{'FPDU':'proton','FHEDU':'heplus','FODU':'oplus'},
                           'energy_range_ev_per_q':[30.,30000.]}}
        field=np.tile([10.,20.,30.],(5,1))
        clean=write_field(root/'clean.cdf',[0,4,8,12,16],field,'64hz')
        clean['source']=fingerprint(root/'clean.cdf')
        request.update(mgf=[clean],output=str(root/'clean.npz'))
        assert worker.execute(request)['status']=='complete'
        with np.load(root/'clean.npz',allow_pickle=False) as data:
            baseline={key:data[key] for key in data.files}
        for var in ('FPDU','FHEDU','FODU'):
            assert np.all(baseline[var+'__density'] > 0), 'Synthetic energy grid must give nonzero moments'
            assert np.isfinite(baseline[var+'__velocity']).all()
        bad=field.copy();bad[2]=-1e31
        staged=write_field(root/'bad.cdf',[0,4,8,12,16],bad,'64hz');staged['source']=fingerprint(root/'bad.cdf')
        request.update(mgf=[staged],output=str(root/'partial.npz'))
        result=worker.execute(request)
        assert result['status']=='partial' and result['computed_records_per_species']==4,result
        assert result['invalid_records_per_species']==1
        with np.load(root/'partial.npz',allow_pickle=False) as data:
            valid=data['moment_valid'];np.testing.assert_array_equal(valid,[True,True,False,True,True])
            np.testing.assert_array_equal(data['moment_invalid_reason'],[0,0,2,0,0])
            np.testing.assert_array_equal(data['epoch_tt2000'],baseline['epoch_tt2000'])
            for var in ('FPDU','FHEDU','FODU'):
                for key in worker.pilot.MOMENTS:
                    name=var+'__'+key
                    assert np.isnan(data[name][~valid]).all(),name
                    np.testing.assert_array_equal(data[name][valid],baseline[name][valid],err_msg=name)
                np.testing.assert_array_equal(data[var+'__counts_raw'],baseline[var+'__counts_raw'])
                np.testing.assert_array_equal(data[var+'__quality_raw'],baseline[var+'__quality_raw'])
                assert np.isnan(data[var+'__b_dsi_at_moment'][~valid]).all()
                np.testing.assert_allclose(data[var+'__b_dsi_at_moment'][valid],baseline[var+'__b_dsi_at_moment'][valid])
            manifest=json.loads(str(data['manifest_json']))
            assert manifest['metadata']['moment_selection']['invalid_reason_counts']['nonfinite_mgf_bracket']==1
        # Even one eligible timestamp is usable, including an exact finite sample
        # surrounded by FILLVAL. Public interpolation must not propagate NaNs.
        singleton=np.full((5,3),-1e31);singleton[2]=[10.,20.,30.]
        staged=write_field(root/'single.cdf',[0,4,8,12,16],singleton,'64hz');staged['source']=fingerprint(root/'single.cdf')
        request.update(mgf=[staged],output=str(root/'single.npz'))
        result=worker.execute(request)
        assert result['status']=='partial' and result['computed_records_per_species']==1,result
        with np.load(root/'single.npz',allow_pickle=False) as data:
            np.testing.assert_array_equal(data['moment_valid'],[False,False,True,False,False])
            for var in ('FPDU','FHEDU','FODU'):
                for key in worker.pilot.MOMENTS:
                    np.testing.assert_array_equal(data[var+'__'+key][2],baseline[var+'__'+key][2],err_msg=var+'__'+key)
        # Coverage, excessive gap and zero interpolated field get separate bits.
        lo=np.datetime64('2022-09-29T05:05:00','ns');hi=lo+np.timedelta64(20,'s');t0=lo.astype('int64')/1e9
        staged=write_field(root/'coverage.cdf',[0,4,12,16],[[1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]],'64hz')
        arrays,_=worker.magnetic([staged],lo,hi,np.array([t0-1,t0,t0+2,t0+6,t0+16,t0+20]),'64hz',4.)
        np.testing.assert_array_equal(arrays['moment_invalid_reason'],[1,0,8,4,0,1])
        assert np.isnan(arrays['mgf_bracket_gap_seconds'][[0,-1]]).all()
        # All-invalid chunks still produce no moment product and retain classification.
        request.update(mgf=[write_field(root/'allbad.cdf',[0,4,8,12,16],np.full((5,3),-1e31),'64hz')],
                       output=str(root/'allbad.npz'))
        request['mgf'][0]['source']=fingerprint(root/'allbad.cdf')
        try:worker.execute(request)
        except worker.MissingMagneticData as exc:
            assert worker.magnetic_failure_result(request,exc)['status']=='excluded'
        else:raise AssertionError('All-invalid chunk was accepted')
        assert not (root/'allbad.npz').exists()
    print('Partial worker passed: all 48 moments match valid reference; full timeline/raw diagnostics; singleton; reason bits; all-invalid guard')


if __name__=='__main__':main()
