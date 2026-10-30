"""Prepare full-range offline calculation from local acquisition receipts; no requests or moments."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import math
from pathlib import Path
import sys

from batch import DEFAULT_OUTPUT, atomic_json, chunks, fingerprint, profile, requested_range
from recovery_plan import CONDITIONS, SCHEMA, digest, required_units


def local_entry(root, u):
    rows=[]
    for category in ('receipts','state'):
        path=root/'verified_inputs'/category/u['date'][:7].replace('-','/')/(u['key']+'.json')
        if path.exists():
            row=json.loads(path.read_text())
            if row.get('unit')==u:
                rows.append(row)
    for row in rows:
        if row.get('status')=='verified':
            try:
                stamp=fingerprint(row['path'])
            except OSError:
                continue
            if stamp==row['fingerprint']:
                return {'status':'local_candidate','unit':u,'path':row['path'],
                        'fingerprint':stamp,'sha256':row['sha256']}
    for row in rows:
        if row.get('transport_length_verified') and row.get('rejected_path') and row.get('rejected_sha256'):
            try:
                stamp=fingerprint(row['rejected_path'])
            except OSError:
                continue
            return {'status':'local_candidate','unit':u,'path':row['rejected_path'],
                    'fingerprint':stamp,'sha256':row['rejected_sha256'],'origin':'previously_rejected_cdf'}
    for row in rows:
        if row.get('status')=='remote_absent':
            evidence={k:row['evidence'].get(k) for k in ('url','http_status','index_sha256')}
            if evidence.get('url') and (evidence.get('index_sha256') or evidence.get('http_status') in (404,410)):
                return {'status':'remote_absent','unit':u,'evidence':evidence}
    raise ValueError('No unchanged local input or proven-absence evidence: '+u['key'])


def prepare(args,start,end):
    identity,config=profile(args)
    intervals=list(chunks(start,end))
    needed=required_units(intervals,args.mgf_cadence)
    def read(pair):
        key,u=pair
        try:
            return key,local_entry(args.output_root,u),None
        except (OSError,ValueError,KeyError) as exc:
            return key,None,str(exc)
    entries={};errors={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for key,entry,error in pool.map(read,needed.items()):
            if error:
                errors[key]=error
            else:
                entries[key]=entry
    if errors:
        preview='; '.join(list(errors.values())[:5])
        raise ValueError(f'{len(errors)} unresolved input units; plan not replaced. '+preview)
    plan={'schema':SCHEMA,'source_profile':str(args.output_root/('native_dsi_'+identity[:12])),
          'source_profile_hash':identity,'conditions':{k:config[k] for k in CONDITIONS},
          'scope':'Full requested range, local inputs only; no archived moment reuse',
          'ranges':[{'start_utc':lo.isoformat(),'end_utc_exclusive':hi.isoformat()} for lo,hi in intervals],
          'inputs':entries}
    plan['plan_sha256']=digest(plan)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--year',type=int);selection.add_argument('--month');selection.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--output-plan',type=Path)
    p.add_argument('--mgf-cadence',choices=['64hz','8sec'],default='64hz')
    p.add_argument('--max-mgf-gap-seconds',type=float)
    p.add_argument('--lepi-boundary-tolerance-seconds',type=float,default=64.)
    p.add_argument('--workers',type=int,default=16,help='Local receipt read workers; no downloads')
    args=p.parse_args()
    if args.workers<1:p.error('--workers must be positive')
    if args.max_mgf_gap_seconds is None:
        args.max_mgf_gap_seconds=1. if args.mgf_cadence=='64hz' else 16.
    if not all(math.isfinite(v) for v in (args.max_mgf_gap_seconds,args.lepi_boundary_tolerance_seconds)):
        p.error('Limits must be finite')
    if args.max_mgf_gap_seconds<=0 or args.lepi_boundary_tolerance_seconds<0:
        p.error('Invalid interpolation/boundary limits')
    try:
        start,end=requested_range(args)
        plan=prepare(args,start,end)
    except (OSError,ValueError,KeyError) as exc:
        print(str(exc),file=sys.stderr);return 1
    if args.output_plan is None:
        label=str(args.year) if args.year is not None else args.month or start.strftime('%Y%m%d_%H%M%S')
        args.output_plan=args.output_root/'plans'/(label+'_full_local.json')
    atomic_json(args.output_plan,plan)
    print(json.dumps({'plan':str(args.output_plan),'chunks':len(plan['ranges']),
                     'inputs':dict(Counter(x['status'] for x in plan['inputs'].values())),
                     'note':'No requests or moment computation; SHA256 and full CDF validation occur during calculation'},indent=2))
    return 0


if __name__=='__main__':
    sys.exit(main())
