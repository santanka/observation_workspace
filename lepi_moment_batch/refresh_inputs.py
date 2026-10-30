"""Explicit product-selective input update. Downloads/validates CDFs; never computes moments."""
import argparse
from collections import Counter
import datetime as dt
import fcntl
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import tempfile

from acquisition import Acquisition, REMOTE, unit
from batch import DEFAULT_INPUT, DEFAULT_OUTPUT, atomic_json, requested_range

PRODUCTS = {'lepi':'lepi', 'mgf64hz':'64hz', 'mgf8sec':'8sec'}


def planned_units(start, end, products):
    planned = {}
    for name in products:
        product = PRODUCTS[name]
        if product == 'lepi':
            cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
            if cursor == start:
                cursor -= dt.timedelta(days=1)
            stop = end if end.time() == dt.time(0) else end-dt.timedelta(microseconds=1)
            step = dt.timedelta(days=1)
        else:
            pad = dt.timedelta(seconds=16 if product == '64hz' else 32)
            cursor = (start-pad).replace(minute=0, second=0, microsecond=0)
            if product == '8sec':
                cursor = cursor.replace(hour=0)
            stop = end+pad
            step = dt.timedelta(hours=1) if product == '64hz' else dt.timedelta(days=1)
        while cursor <= stop:
            u = unit(product,cursor)
            planned[u['key']] = (product,cursor)
            cursor += step
    return sorted(planned.values(),key=lambda p:(p[1],p[0]))


def run(args, planned):
    from tqdm import tqdm
    args.output_root.mkdir(parents=True,exist_ok=True)
    args.scratch_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'.input_refresh.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='lepi_input_update_',dir=args.scratch_root) as directory:
            scratch = Path(directory)
            acquisition = Acquisition(args,scratch)
            enabled = args.progress=='on' or (args.progress=='auto' and sys.stderr.isatty())
            with tqdm(total=len(planned),desc='Update inputs',unit='file',disable=not enabled) as bar:
                if args.download_workers>1:
                    statuses = acquisition.prefetch_parallel(planned,bar,args.download_workers)
                else:
                    statuses = {}
                    for product,date in planned:
                        try:
                            row = acquisition.ensure(product,date)
                            statuses[row['unit']['key']] = row['status']
                            bar.update(1)
                        finally:
                            shutil.rmtree(scratch/'cdf',ignore_errors=True)
            report = {'products':args.products,'counts':dict(Counter(statuses.values())),
                      'units':statuses,'note':'Input update only; no moment computation'}
            atomic_json(args.output_root/'verified_inputs/last_product_refresh.json',report)
            print(json.dumps({k:v for k,v in report.items() if k!='units'}),flush=True)
            return 0 if all(s in ('verified','remote_absent') for s in statuses.values()) else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--year',type=int)
    selection.add_argument('--month')
    selection.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--products',nargs='+',choices=PRODUCTS,required=True)
    p.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--scratch-root',type=Path,default=Path('/tmp/lepi_input_updates'))
    p.add_argument('--download-workers',type=int,default=8)
    p.add_argument('--progress',choices=['auto','on','off'],default='auto')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--remote-root',default=REMOTE)
    p.add_argument('--min-free-memory-gib',type=float,default=8.)
    p.add_argument('--scratch-reserve-gib',type=float,default=4.)
    p.add_argument('--download-attempts',type=int,default=3)
    p.add_argument('--retry-delay',type=float,default=30.)
    p.add_argument('--http-timeout',type=float,default=60.)
    p.add_argument('--download-timeout',type=float,default=600.)
    p.add_argument('--validation-timeout',type=float,default=180.)
    p.set_defaults(offline=False,refresh_downloads=True,max_network_failures=0,
                   lepi_boundary_tolerance_seconds=64.)
    return p


def main():
    p=parser();args=p.parse_args()
    if not all(math.isfinite(v) for v in (args.http_timeout,args.download_timeout,args.validation_timeout,args.min_free_memory_gib,args.scratch_reserve_gib,args.retry_delay)):
        p.error('Limits must be finite')
    if args.download_workers<1 or args.download_attempts<1 or min(args.http_timeout,args.download_timeout,args.validation_timeout)<=0:
        p.error('Invalid download limits')
    if min(args.min_free_memory_gib,args.scratch_reserve_gib,args.retry_delay)<0:
        p.error('Invalid resource limits')
    try:
        start,end=requested_range(args)
        planned=planned_units(start,end,args.products)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        print(json.dumps({'start_utc':start.isoformat(),'end_utc_exclusive':end.isoformat(),
             'products':args.products,'granules':dict(Counter(p for p,_ in planned)),
             'note':'Boundary granules included; no requests or writes'},indent=2))
        return 0
    def stop(signum,frame):
        raise KeyboardInterrupt(f'Stop signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        return run(args,planned)
    except KeyboardInterrupt:
        return 130
    except (OSError,RuntimeError,ValueError) as exc:
        print(str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
