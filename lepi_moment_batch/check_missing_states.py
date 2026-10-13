"""End-to-end tiny CDF acquisition over loopback HTTP, then batch state/resume guards."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from test_acquisition import fixture

BASE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory(prefix='lepi_acquisition_') as directory:
        root = Path(directory)
        path = root/'fixture.cdf'
        fixture(path, product='lepi')
        body = path.read_bytes()
        name = 'erg_lepi_l2_3dflux_20220929_v03_00.cdf'
        fail_all = False
        fail_once = False
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal fail_once
                if fail_once and self.path.endswith("/" + name):
                    fail_once = False
                    self.send_error(503)
                    return
                if fail_all:
                    self.send_error(503)
                    return
                if self.path.endswith('/'):
                    listing = '<html><h1>Index of ' + self.path.rstrip('/') + '</h1>'
                    if '/lepi/' in self.path:
                        listing += '<a href="' + name + '">CDF</a>'
                    data = (listing + '</html>').encode()
                elif self.path.endswith('/' + name):
                    data = body
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = [sys.executable, str(BASE/'batch.py'), '--input-root', str(root/'raw'),
                    '--output-root', str(root/'output'), '--scratch-root', str(root/'scratch'),
                    '--remote-root', f'http://127.0.0.1:{server.server_port}/erg/',
                    '--pause-seconds', '0', '--retry-delay', '0', '--http-timeout', '2',
                    '--download-attempts', '2', '--retry-pass-delay', '0', '--progress', 'on']
            def run(start, end, expected, code, extra=()):
                result = subprocess.run(base+['--start', start, '--end', end]+list(extra),
                                        capture_output=True, text=True, timeout=45)
                assert result.returncode == code, result.stderr + result.stdout
                assert '100%' in result.stderr and '1/1' in result.stderr, result.stderr
                report = json.loads(result.stdout.splitlines()[-1])
                assert report['counts'] == {expected: 1}, report
                return report
            run('2022-09-29T05:05', '2022-09-29T05:10', 'missing_mgf', 2)
            report = run('2022-09-29T05:10', '2022-09-29T05:15', 'no_data', 0)
            run('2022-09-29T05:10', '2022-09-29T05:15', 'skipped_verified', 0)
            run('2022-09-28T05:05', '2022-09-28T05:10', 'remote_absent', 0)
            run('2022-09-29T05:05', '2022-09-29T05:10', 'timeout', 2,
                ['--worker-timeout', '0.01'])
            assert not list((root/'scratch').glob('*/cdf/**/*'))
            verified = list((root/'raw').rglob('*.cdf'))
            assert len(verified) == 1 and verified[0].read_bytes() == body
            state = next((root/'output/verified_inputs/state').glob('*/*/lepi_20220929.json'))
            assert json.loads(state.read_text())['origin'] == 'http_download'
            # Lose a required LEP-i variable: even with complete transport, no_data must be refused.
            fixture(path, product='lepi', omit='FODU_Quality')
            body = path.read_bytes()
            run('2022-09-29T05:10', '2022-09-29T05:15', 'failed_validation', 2,
                ['--refresh-downloads'])
            # Only failed download units are fetched again after the first calculation pass.
            fixture(path, product='lepi')
            body = path.read_bytes()
            fail_once = True
            recovered = run('2022-09-29T05:10', '2022-09-29T05:15', 'no_data', 0,
                            ['--refresh-downloads', '--download-attempts', '1'])
            assert not fail_once
            latest = json.loads((Path(recovered['output_root'])/'last_run.json').read_text())
            assert latest['status'] == 'finished' and latest['counts'] == {'no_data': 1}, latest
            # A sustained outage stops the whole request, records current counts, and leaves no NPZ.
            fail_all = True
            stopped = subprocess.run(base + ['--start', '2021-09-29T05:05', '--end', '2021-10-02T05:05',
                                             '--download-attempts', '1', '--max-network-failures', '3'],
                                     capture_output=True, text=True, timeout=90)
            assert stopped.returncode == 1, stopped.stderr + stopped.stdout
            latest = json.loads((Path(report['output_root'])/'last_run.json').read_text())
            assert latest['status'] == 'stopped' and 'Stopping after 3' in latest['reason'], latest
            assert latest['counts'] == {}, latest  # circuit breaker fires in acquisition before any calculation
            failures = [json.loads(p.read_text()) for p in (root/'output/verified_inputs/state/2021').glob('*/*.json')]
            assert sum(x['status'] == 'failed_download' for x in failures) == 3
            assert not list((Path(report['output_root'])/'2021').glob('*/data/*/*.npz'))
        finally:
            server.shutdown()
            server.server_close()
    print('HTTP/CDF integration passed: download, missing MGF, verified no_data, resume, '
          'remote absence, timeout, cleanup, invalid refreshed input blocks no_data, sustained outage stops run')


if __name__ == '__main__':
    main()
