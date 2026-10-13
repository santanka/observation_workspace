"""Compare one saved GSM chunk with public erg_cotrans + cotrans, fully offline."""
import argparse
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import convert


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.output,allow_pickle=False) as data:
        manifest=json.loads(str(data['manifest_json']))
        targets=data['time_unix'].copy(); matrices=data['rotation_dsi_to_gsm'].copy()
        observed={s:data[s+'__velocity_gsm_observed'].copy() for s in convert.SPECIES}
        masks={s:data[s+'__velocity_valid'].copy() for s in convert.SPECIES}
    config=manifest['config']
    support=convert.Support(SimpleNamespace(offline=True,download_attempts=1,retry_delay=0,
                                            earth_radius_km=config['earth_radius_km']))
    day=convert.parse(manifest['start_utc']).replace(hour=0,minute=0,second=0,microsecond=0)
    support.ensure(day)
    import pyspedas as psp
    from pyspedas.cotrans_tools.cotrans import cotrans
    from pyspedas.projects.erg.satellite.erg.common.cotrans.erg_cotrans import erg_cotrans
    # Independent interpolation for reference. Public DSI routine uses cached sun_j2000
    # when noload=True. Supply the satellite-to-Sun direction with consistent km units.
    position=np.column_stack([np.interp(targets,support.ot,support.positions[:,j]) for j in range(3)])
    sundir=np.array([1.496e8,0.,0.])-position
    sundir/=np.linalg.norm(sundir,axis=1)[:,None]
    sj=cotrans(time_in=targets,data_in=sundir,coord_in='gse',coord_out='j2000',position=False,quiet=True)
    psp.store_data('sundir_j2000',data={'x':targets,'y':sj})
    results={}
    source=Path(manifest['inputs'][0]['fingerprint']['path'])
    before=convert.sha(source)
    with np.load(source,allow_pickle=False) as native:
        np.testing.assert_array_equal(targets,native['time_unix'])
        for species in convert.SPECIES:
            velocity=native[species+'__velocity']
            psp.store_data('reference_dsi',data={'x':targets,'y':velocity})
            erg_cotrans(in_name='reference_dsi',out_name='reference_j2000',
                        in_coord='dsi',out_coord='j2000',noload=True)
            reference=cotrans(time_in=targets,data_in=psp.get_data('reference_j2000').y,
                              coord_in='j2000',coord_out='gsm',position=False,quiet=True)
            good=masks[species]
            np.testing.assert_allclose(observed[species][good],reference[good],rtol=1e-11,atol=1e-9)
            recovered=np.einsum('nji,nj->ni',matrices[good],observed[species][good])
            np.testing.assert_allclose(recovered,velocity[good],rtol=1e-11,atol=1e-9)
            results[species]={'finite_records':int(good.sum()),
                             'max_abs_reference_error_kms':float(np.max(np.abs(observed[species][good]-reference[good]))) if good.any() else None,
                             'max_abs_inverse_error_kms':float(np.max(np.abs(recovered-velocity[good]))) if good.any() else None}
    assert before==convert.sha(source)
    report={'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'output':str(args.output),
            'output_sha256':convert.sha(args.output),'source_unchanged':True,
            'reference':'Public erg_cotrans DSI->J2000 + cotrans J2000->GSM; cached sun direction uses orbit RE converted to km',
            'velocity_correction_applied':False,'records':len(targets),'species':results}
    convert.atomic_json(args.report,report)
    print(json.dumps(report))


if __name__=='__main__': main()
