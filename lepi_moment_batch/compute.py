"""Bounded, isolated five-minute computation; only the parent records results."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

from batch import BASE, atomic_json


class ComputePool:
    def __init__(self, args, finished):
        self.args, self.finished = args, finished
        self.workers = getattr(args, 'compute_workers', 1)
        self.cpus = queue.Queue()
        allowed = getattr(args, 'compute_cpu_ids', sorted(os.sched_getaffinity(0)))
        if self.workers > len(allowed):
            raise ValueError('compute-workers exceeds available CPU count')
        for cpu in allowed[-self.workers:]:
            self.cpus.put(cpu)
        self.cancelled = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        self.pending = {}

    def __enter__(self):
        return self

    def collect(self, all_results=False):
        while self.pending:
            ready, _ = wait(self.pending, return_when=FIRST_COMPLETED)
            for future in ready:
                common, state_path, label = self.pending.pop(future)
                state = {**common, **future.result()}
                atomic_json(state_path, state)
                self.finished(label, state)
            if not all_results:
                break

    def wait_for_slot(self):
        if len(self.pending) >= self.workers:
            self.collect()

    def submit(self, label, request, common, state_path, job, log):
        self.wait_for_slot()
        job.mkdir(parents=True, exist_ok=True)
        request_path, result_path = job/'request.json', job/'result.json'
        atomic_json(request_path, request)
        result_path.unlink(missing_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(state_path, {**common, 'status': 'running'})
        future = self.executor.submit(self.execute, request_path, result_path, log)
        self.pending[future] = (common, state_path, label)

    def execute(self, request, result, log):
        cpu = self.cpus.get()
        proc = None
        state = None
        try:
            if self.cancelled.is_set():
                return {'status': 'interrupted', 'failure_class': 'cancelled', 'retryable': True, 'log': str(log)}
            env = {**os.environ, 'LEPI_COMPUTE_CPU': str(cpu)}
            with log.open('w') as stream:
                proc = subprocess.Popen([sys.executable, str(BASE/'worker.py'),
                                          '--request', str(request), '--result', str(result)],
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=env)
                import time
                deadline = time.monotonic() + self.args.worker_timeout
                while proc.poll() is None:
                    if self.cancelled.is_set():
                        state = {'status': 'interrupted', 'failure_class': 'cancelled', 'retryable': True}
                        break
                    if time.monotonic() >= deadline:
                        state = {'status': 'timeout', 'failure_class': 'compute_timeout', 'retryable': True}
                        break
                    self.cancelled.wait(min(.05, max(0., deadline-time.monotonic())))
                if state is None:
                    if proc.returncode or not result.exists():
                        state = {'status': 'failed', 'failure_class': 'worker_exit',
                                 'retryable': True, 'returncode': proc.returncode}
                    else:
                        state = json.loads(result.read_text())
            if state['status'] in {'complete', 'partial'}:
                self.cancelled.wait(self.args.pause_seconds)
            return {**state, 'log': str(log)}
        except Exception as exc:
            return {'status': 'failed', 'failure_class': 'worker_orchestration_error', 'retryable': True,
                    'reason': str(exc), 'exception_type': type(exc).__name__, 'log': str(log)}
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            self.cpus.put(cpu)

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.cancelled.set()
        try:
            self.collect(all_results=True)
        except BaseException:
            self.cancelled.set()
            self.executor.shutdown(wait=True)
            self.collect(all_results=True)
            raise
        finally:
            self.executor.shutdown(wait=True)
