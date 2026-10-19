"""Bounded scheduling, failure stopping and cross-process validation exclusion."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from acquisition import Acquisition, AcquisitionError, unit
from tqdm import tqdm


class DownloadWorkerTests(unittest.TestCase):
    def test_prefetch_dispatch_and_completion_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(output_root=root/'out', input_root=root/'raw',
                    remote_root='http://127.0.0.1/erg/', offline=False,
                    download_workers=8, max_network_failures=0)
            acquisition = Acquisition(args, root)
            guard = threading.Lock()
            gate = threading.Event()
            active = peak = 0
            def acquire(product, date, catalogs, cancelled):
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                    if active == 8:
                        gate.set()
                assert gate.wait(5.)
                with guard:
                    active -= 1
                return {'result': {'unit': unit(product, date), 'status': 'remote_absent'}, 'catalogs': {}}
            day = dt.datetime(2022, 9, 29, tzinfo=dt.timezone.utc)
            with patch.object(acquisition, 'acquire_isolated', side_effect=acquire) as child:
                states = acquisition.prefetch([(day, day, day + dt.timedelta(hours=8))], '64hz', 'off')
                self.assertEqual(peak, 8)
                self.assertEqual(len(states), child.call_count)
                self.assertEqual(len(states), len(acquisition.results))
                before = child.call_count
                self.assertEqual(acquisition.prefetch([(day, day, day + dt.timedelta(hours=8))], '64hz', 'off'), states)
                self.assertEqual(child.call_count, before)

    def test_circuit_breaker_cancels_inflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(output_root=root/'out', input_root=root/'raw',
                    remote_root='http://127.0.0.1/erg/', offline=False, max_network_failures=1)
            acquisition = Acquisition(args, root)
            first = threading.Event()
            cancelled_jobs = []
            def acquire(product, date, catalogs, cancelled):
                if not first.is_set():
                    first.set()
                    return {'result': {'unit': unit(product, date), 'status': 'failed_download'}, 'catalogs': {}}
                assert cancelled.wait(5.), 'Other in-flight workers were not cancelled'
                cancelled_jobs.append(True)
                raise InterruptedError('cancelled')
            day = dt.datetime(2022, 9, 29, tzinfo=dt.timezone.utc)
            planned = [('64hz', day + dt.timedelta(hours=h)) for h in range(20)]
            with patch.object(acquisition, 'acquire_isolated', side_effect=acquire) as child:
                with tqdm(total=20, disable=True) as bar:
                    with self.assertRaisesRegex(AcquisitionError, 'Stopping after 1'):
                        acquisition.prefetch_parallel(planned, bar, 8)
                self.assertLessEqual(child.call_count, 8)
                self.assertTrue(cancelled_jobs)

    def test_validation_slot_is_exclusive_across_processes(self):
        script = '''
import argparse, json, os, sys, time
from pathlib import Path
from acquisition import Acquisition
from acquisition_worker import SerialValidationAcquisition
root=Path(sys.argv[1])
a=SerialValidationAcquisition(argparse.Namespace(output_root=root/'out', input_root=root/'raw',
       remote_root='http://127.0.0.1/erg/', validation_lock=str(root/'slot.lock')),root)
def fake(self,path,u):
    with (root/'events').open('a') as out: out.write('start '+str(os.getpid())+'\\n')
    time.sleep(.1)
    with (root/'events').open('a') as out: out.write('end '+str(os.getpid())+'\\n')
    return {}, {}
Acquisition.verify=fake
try: a.verify(None,None)
finally: a.release_validation()
'''
        with tempfile.TemporaryDirectory() as directory:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(subprocess.run, [sys.executable, '-c', script, directory],
                           cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=15) for _ in range(4)]
                for future in futures:
                    result = future.result()
                    self.assertEqual(result.returncode, 0, result.stderr)
            events = (Path(directory)/'events').read_text().splitlines()
            self.assertEqual(len(events), 8)
            for start, end in zip(events[::2], events[1::2]):
                self.assertEqual(start.split()[0], 'start')
                self.assertEqual(end.split()[0], 'end')
                self.assertEqual(start.split()[1], end.split()[1])


if __name__ == '__main__':
    unittest.main()
