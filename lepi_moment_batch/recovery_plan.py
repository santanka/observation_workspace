"""Frozen, local-only recovery selection and input provenance. No network access."""
import datetime as dt
import hashlib
import json
from pathlib import Path

SCHEMA = 'lepi_local_recovery_plan_v1'
CONDITIONS = ('energy_range_ev_per_q', 'species', 'frame', 'spacecraft_potential_volts_used',
              'quality_mask_applied', 'lepi_boundary_tolerance_seconds', 'mgf_cadence', 'max_mgf_gap_seconds')


def digest(plan):
    return hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k != 'plan_sha256'}, sort_keys=True).encode()).hexdigest()


def ranges(plan):
    result = []
    for row in plan['ranges']:
        points = [dt.datetime.fromisoformat(row[k]) for k in ('start_utc','end_utc_exclusive')]
        if any(p.tzinfo is None for p in points):
            raise ValueError('Recovery timestamps require an explicit timezone')
        lo,hi = (p.astimezone(dt.timezone.utc) for p in points)
        if hi-lo != dt.timedelta(minutes=5) or any(p.second or p.microsecond or p.minute%5 for p in (lo,hi)):
            raise ValueError('Recovery ranges must be UTC five-minute chunks')
        result.append((lo,hi))
    if not result or len(set(result)) != len(result):
        raise ValueError('Empty or duplicate recovery ranges')
    return sorted(result)


def load(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('schema') != SCHEMA or plan.get('plan_sha256') != digest(plan):
        raise ValueError('Invalid recovery plan identity')
    ranges(plan)
    for key,entry in plan['inputs'].items():
        if entry.get('status') not in ('local_candidate','remote_absent') or entry['unit']['key'] != key:
            raise ValueError('Invalid recovery input entry: '+key)
    return plan


def required_units(intervals, cadence):
    from acquisition import unit
    result = {}
    for lo,hi in intervals:
        day = lo.replace(hour=0,minute=0,second=0,microsecond=0)
        dates = [day]
        if lo == day: dates.insert(0,day-dt.timedelta(days=1))
        if hi == day+dt.timedelta(days=1): dates.append(day+dt.timedelta(days=1))
        for date in dates:
            u=unit('lepi',date);result[u['key']]=u
        cursor=(lo-dt.timedelta(seconds=16 if cadence=='64hz' else 32)).replace(minute=0,second=0,microsecond=0)
        if cadence=='8sec': cursor=cursor.replace(hour=0)
        while cursor <= hi+dt.timedelta(seconds=16 if cadence=='64hz' else 32):
            u=unit(cadence,cursor);result[u['key']]=u
            cursor+=dt.timedelta(hours=1) if cadence=='64hz' else dt.timedelta(days=1)
    return result


def prepare(audit_path, baseline, output_root):
    from batch import fingerprint, sha_file
    states=json.loads(Path(audit_path).read_text())
    source=json.loads((Path(baseline)/'profile.json').read_text())
    selected=[]; candidates={}
    for state in states:
        reason=state.get('reason','')
        if ((state.get('status')=='failed' and state.get('exception_type')=='InvalidIndexError')
            or (state.get('status')=='failed_validation' and
                ('Conflicting records at a duplicate epoch:' in reason or 'Epochs outside nominal granule' in reason))):
            selected.append({k:state[k] for k in ('start_utc','end_utc_exclusive','state_path','status','reason')})
        for entry in state.get('acquisition_evidence',[]):
            if entry['status']=='verified':
                candidates[entry['unit']]={'status':'local_candidate','path':entry['fingerprint']['path'],
                    'fingerprint':entry['fingerprint'],'sha256':entry['sha256']}
            elif entry['status']=='remote_absent' and entry['unit'] not in candidates:
                candidates[entry['unit']]={'status':'remote_absent','evidence':entry['absence_evidence']}
        for failed in state.get('failed_units',[]):
            if failed.get('transport_length_verified') and failed.get('rejected_path'):
                path=Path(failed['rejected_path'])
                candidates[failed['unit']['key']]={'status':'local_candidate','path':str(path),
                    'fingerprint':fingerprint(path),'sha256':failed['rejected_sha256'],
                    'origin':'previously_rejected_cdf'}
    plan={'schema':SCHEMA,'source_profile':str(Path(baseline).resolve()),'source_profile_hash':source['profile_hash'],
          'audit_path':str(Path(audit_path).resolve()),'audit_sha256':sha_file(audit_path),
          'conditions':{k:source['config'][k] for k in CONDITIONS},'ranges':selected,'inputs':{}}
    for key,u in required_units(ranges(plan),source['config']['mgf_cadence']).items():
        candidate=candidates.get(key)
        if candidate is None:
            # Local acquisition receipts can supply a boundary unit omitted by a failed primary.
            for category in ('receipts','state'):
                path=Path(output_root)/'verified_inputs'/category/u['date'][:7].replace('-','/')/(key+'.json')
                if not path.exists(): continue
                row=json.loads(path.read_text())
                if row.get('status')=='verified':
                    candidate={'status':'local_candidate','path':row['path'],'fingerprint':row['fingerprint'],'sha256':row['sha256']}
                elif row.get('status')=='remote_absent':
                    candidate={'status':'remote_absent','evidence':row['evidence']}
                if candidate: break
        if candidate is None:
            raise ValueError('No local input or proven-absence evidence for '+key)
        if candidate['status']=='local_candidate' and fingerprint(candidate['path']) != candidate['fingerprint']:
            raise ValueError('Local input fingerprint changed: '+key)
        plan['inputs'][key]={**candidate,'unit':u}
    plan['plan_sha256']=digest(plan)
    return plan
