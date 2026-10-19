"""Loopback-only integration for eight acquisition workers and interruption cleanup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import tempfile
import threading
import time

from acquisition import Acquisition
from test_acquisition import fixture
from tqdm import tqdm


def main():
    with tempfile.TemporaryDirectory(prefix='lepi_parallel_check_') as directory:
        root = Path(directory)
        start = dt.datetime(2022, 9, 29, tzinfo=dt.timezone.utc)
        files = {}
        for hour in range(8):
            p = root / f'fixture{hour}.cdf'
            fixture(p, date=(2022, 9, 29, hour, 5))
            files[f'erg_mgf_l2_64hz_dsi_20220929{hour:02d}_v04.06.cdf'] = p.read_bytes()
        lock = threading.Lock()
        active = peak = gets = 0
        release = threading.Event()
        slow = False
        started = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal active, peak, gets
                if self.path.endswith('/'):
                    listing = '<html><h1>Index of ' + self.path.rstrip('/') + '</h1>'
                    body = (listing + ''.join(f'<a href="{name}">CDF</a>' for name in files) + '</html>').encode()
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                name = self.path.rsplit('/', 1)[-1]
                if name not in files:
                    self.send_error(404)
                    return
                with lock:
                    active += 1
                    gets += 1
                    peak = max(peak, active)
                    if active == 8:
                        release.set()
                started.set()
                try:
                    if slow:
                        release.wait(10.)
                    else:
                        release.wait(5.)
                    body = files[name]
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    with lock:
                        active -= 1
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        args = argparse.Namespace(output_root=root/'out', input_root=root/'raw',
                remote_root=f'http://127.0.0.1:{server.server_port}/erg/', offline=False,
                refresh_downloads=False, download_attempts=1, retry_delay=0., http_timeout=15.,
                download_timeout=20., validation_timeout=20., min_free_memory_gib=0.,
                scratch_reserve_gib=0., download_workers=8, max_network_failures=0,
                reuse_results_from=[root/'first_profile',root/'second_profile'])
        scratch = root/'scratch'
        scratch.mkdir()
        acquisition = Acquisition(args, scratch)
        planned = [('64hz', start + dt.timedelta(hours=hour)) for hour in range(8)]
        try:
            with tqdm(total=8, disable=True) as bar:
                states = acquisition.prefetch_parallel(planned, bar, 8)
            assert peak == 8, peak
            assert gets == 8, gets
            assert len(states) == 8 and set(states.values()) == {'verified'}, states
            assert len(acquisition.results) == 8
            assert not list((scratch/'prefetch').glob('*/cdf'))
            for entry in acquisition.results.values():
                assert entry['validation_identity'] == acquisition.validation_identity
                assert Path(entry['path']).read_bytes() == files[Path(entry['path']).name]
            # Cached in-memory results must not launch more requests.
            with tqdm(total=8, disable=True) as bar:
                again = acquisition.prefetch_parallel(planned, bar, 8)
            assert again == states and gets == 8
            # Stop while a worker is blocked in HTTP; the parent must terminate it promptly.
            args.refresh_downloads = True
            slow = True
            release.clear()
            started.clear()
            def interrupt():
                assert started.wait(10.)
                __import__('os').kill(__import__('os').getpid(), signal.SIGUSR1)
            def stopped(signum, frame):
                raise KeyboardInterrupt('test cancellation')
            previous = signal.signal(signal.SIGUSR1, stopped)
            threading.Thread(target=interrupt, daemon=True).start()
            second = Acquisition(args, scratch)
            t0 = time.monotonic()
            try:
                with tqdm(total=1, disable=True) as bar:
                    second.prefetch_parallel(planned[:1], bar, 8)
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError('Interruption was swallowed')
            finally:
                signal.signal(signal.SIGUSR1, previous)
            assert time.monotonic() - t0 < 6., 'Cancellation did not terminate worker promptly'
            assert not list((scratch/'prefetch').glob('*/request.json'))
            assert not list((root/'raw').rglob('*.part'))
            print('Parallel integration passed: 8 simultaneous GETs, 8 verified CDFs, cache reuse, interrupt cleanup')
        finally:
            release.set()
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
