"""Serial verified acquisition into the standard SPEDAS raw-data cache."""
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from batch import BASE, atomic_json, fingerprint, sha_file, stage_cdf, wait_resources

REMOTE = 'https://ergsc.isee.nagoya-u.ac.jp/data/ergsc/satellite/erg/'


class AcquisitionError(RuntimeError):
    def __init__(self, status, reason, retryable=True, **details):
        self.status = status
        self.retryable = retryable
        self.details = details
        super().__init__(reason)


@contextmanager
def wall_timeout(seconds):
    """Bound slowly trickling HTTP responses too (runner executes on the main thread)."""
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise AcquisitionError('failed_download', 'HTTP wall timeout')
    signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, previous)


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.headings = [], []
        self.in_heading = False
    def handle_starttag(self, tag, attrs):
        if tag in ('title', 'h1'):
            self.in_heading = True
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href')
    def handle_endtag(self, tag):
        if tag in ('title', 'h1'):
            self.in_heading = False
    def handle_data(self, data):
        if self.in_heading:
            self.headings.append(data.strip())


def parse_index(body, url):
    parser = IndexParser()
    parser.feed(body.decode('utf-8', errors='replace'))
    expected = 'Index of ' + urllib.parse.unquote(urllib.parse.urlsplit(url).path.rstrip('/'))
    if (not any(x.rstrip('/') == expected for x in parser.headings)
            or not body.rstrip().lower().endswith(b'</html>')):
        raise AcquisitionError('failed_download', 'Unrecognized directory index; absence unconfirmed')
    return [x for x in parser.links if re.fullmatch(r'[A-Za-z0-9_.-]+\.cdf', x)]


def unit(product, date):
    prefix, fmt = ('erg_lepi_l2_3dflux_', '%Y%m%d') if product == 'lepi' else (
        'erg_mgf_l2_64hz_dsi_', '%Y%m%d%H') if product == '64hz' else ('erg_mgf_l2_8sec_', '%Y%m%d')
    folder = ('lepi/l2/3dflux' if product == 'lepi' else 'mgf/l2/' + product) + date.strftime('/%Y/%m')
    return {'product': product, 'date': date.isoformat(), 'folder': folder,
            'prefix': prefix + date.strftime(fmt), 'key': product + '_' + date.strftime(fmt)}


class Acquisition:
    def __init__(self, args, scratch):
        self.args, self.scratch = args, scratch
        self.root = args.output_root / 'verified_inputs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.results, self.catalogs = {}, {}
        self.network_failure_streak = 0
        self.last_http = {}
        self.validator_hash = sha_file(BASE / 'validate_input.py')
        self.validation_identity = hashlib.sha256(json.dumps({
            'validator': self.validator_hash, 'acquisition': sha_file(BASE / 'acquisition.py'),
            'stager': sha_file(BASE / 'batch.py'),
            'cdflib': importlib.metadata.version('cdflib'),
            'numpy': importlib.metadata.version('numpy'),
            'lepi_boundary_tolerance_seconds': getattr(args, 'lepi_boundary_tolerance_seconds', 64.),
            'cdf_sources': {name: sha_file(Path(importlib.metadata.distribution('cdflib').locate_file('cdflib')) / name)
                            for name in ('cdfread.py', 'epochs.py')}
        }, sort_keys=True).encode()).hexdigest()
        self.remote_root = args.remote_root.rstrip('/') + '/'
        parsed = urllib.parse.urlsplit(self.remote_root)
        if parsed.username or parsed.password or parsed.scheme not in ('https', 'http'):
            raise ValueError('remote-root must be HTTP(S), without credentials')
        if parsed.scheme == 'http' and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Unencrypted HTTP is permitted only for loopback tests')

    def request(self, url, method='GET'):
        # Do not silently accept an authentication redirect or a different endpoint.
        self.last_http = {'url': url, 'method': method}
        req = urllib.request.Request(url, method=method, headers={'Accept-Encoding': 'identity',
                                       'User-Agent': 'Arase-LEPi-moment-acquisition/1'})
        response = urllib.request.urlopen(req, timeout=self.args.http_timeout)
        if response.geturl() != url:
            response.close()
            raise AcquisitionError('failed_download', 'Unexpected HTTP redirect')
        if response.status != 200:
            response.close()
            raise AcquisitionError('failed_download', 'Expected HTTP 200')
        return response

    def catalog(self, folder):
        if folder in self.catalogs:
            outcome = self.catalogs[folder]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        url = self.remote_root + folder + '/'
        try:
            with wall_timeout(self.args.http_timeout), self.request(url) as response:
                body = response.read(4 * 1024**2 + 1)
                if len(body) > 4 * 1024**2:
                    raise AcquisitionError('failed_download', 'Directory index exceeds size limit')
                expected_bytes = response.headers.get('Content-Length')
                if expected_bytes is not None and len(body) != int(expected_bytes):
                    raise AcquisitionError('failed_download', 'Truncated directory index')
                names = parse_index(body, url)
                outcome = {'names': names, 'url': url, 'http_status': 200,
                           'index_sha256': hashlib.sha256(body).hexdigest()}
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                outcome = {'names': [], 'url': url, 'http_status': 404}
            else:
                raise
        self.catalogs[folder] = outcome
        return outcome

    def verify(self, path, u):
        cache = self.scratch / 'cdf' / u['date'][:10].replace('-', '')
        try:
            staged = stage_cdf(path, cache, self.args)
        except (OSError, ValueError, EOFError, MemoryError) as exc:
            raise AcquisitionError('failed_validation', 'CDF staging validation: ' + str(exc), phase='stage') from exc
        fd, result = tempfile.mkstemp(prefix='validate_', suffix='.json', dir=self.scratch)
        os.close(fd)
        proc = None
        successful = False
        try:
            proc = subprocess.Popen([sys.executable, str(BASE / 'validate_input.py'),
                    '--path', staged['expanded_path'], '--product', u['product'],
                    '--date', u['date'].replace('+00:00', ''), '--result', result,
                    '--boundary-tolerance-seconds', str(getattr(self.args, 'lepi_boundary_tolerance_seconds', 64.)
                                                      if u['product'] == 'lepi' else 32.)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            try:
                _, error = proc.communicate(timeout=self.args.validation_timeout)
            except BaseException:
                os.killpg(proc.pid, 9)
                proc.wait()
                raise
            if proc.returncode:
                raise AcquisitionError('failed_validation', f'Validator failed with returncode {proc.returncode}: ' + error.decode()[-1000:],
                                       retryable=False, validator_returncode=proc.returncode, phase='validation')
            report = json.loads(Path(result).read_text())
            if report['status'] != 'verified':
                raise AcquisitionError('failed_validation', report['reason'],
                                       retryable=report.get('retryable', True), validator_report=report, phase='validation')
            successful = True
            return staged, report
        except subprocess.TimeoutExpired as exc:
            raise AcquisitionError('failed_validation', 'Validator timeout', retryable=False,
                                   validator_timeout_seconds=self.args.validation_timeout, phase='validation') from exc
        finally:
            Path(result).unlink(missing_ok=True)
            if not successful and staged['compressed']:
                expanded = Path(staged['expanded_path'])
                expanded.unlink(missing_ok=True)
                expanded.with_suffix('.json').unlink(missing_ok=True)

    def publish_staged(self, staged, target):
        """Adopt already expanded bytes after atomic publication; avoid a second daily expansion."""
        if not staged['compressed']:
            return
        expanded = Path(staged['expanded_path'])
        old_marker = expanded.with_suffix('.json')
        fp = fingerprint(target)
        key = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()
        dest = expanded.parent / (key + '.cdf')
        if expanded != dest:
            os.replace(expanded, dest)
            old_marker.unlink(missing_ok=True)
        atomic_json(dest.with_suffix('.json'), {**staged, 'source': fp, 'expanded_path': str(dest)})

    def fetch_candidate(self, url, target, local, u):
        fd, temp = tempfile.mkstemp(prefix=target.name + '.', suffix='.part', dir=target.parent)
        candidate = Path(temp)
        os.close(fd)
        staged = None
        try:
            size, headers, use_local = None, {}, False
            if local and local.is_file():
                try:
                    with wall_timeout(self.args.http_timeout), self.request(url, 'HEAD') as response:
                        size = int(response.headers.get('Content-Length', '0'))
                        headers = dict(response.headers)
                        use_local = size > 0 and local.stat().st_size == size
                except urllib.error.HTTPError as exc:
                    if exc.code not in (405, 501):
                        raise
            if use_local:
                if shutil.disk_usage(target.parent).free < size + self.args.scratch_reserve_gib * 1024**3:
                    raise AcquisitionError('failed_download', 'Insufficient private-cache disk space')
                initial = fingerprint(local)
                with local.open('rb') as inp, candidate.open('wb') as out:
                    shutil.copyfileobj(inp, out, 1024**2)
                    out.flush()
                    os.fsync(out.fileno())
                if fingerprint(local) != initial:
                    raise AcquisitionError('failed_validation', 'Shared raw input changed during copy')
                origin = 'shared_raw_copy_with_remote_length'
            else:
                origin = 'http_download'
                deadline = time.monotonic() + self.args.download_timeout
                with wall_timeout(self.args.download_timeout), self.request(url) as response, candidate.open('wb') as out:
                    size = int(response.headers.get('Content-Length', '0'))
                    headers = dict(response.headers)
                    if size <= 0 or response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise AcquisitionError('failed_download', 'Missing positive Content-Length or encoded response')
                    if shutil.disk_usage(target.parent).free < size + self.args.scratch_reserve_gib * 1024**3:
                        raise AcquisitionError('failed_download', 'Insufficient private-cache disk space')
                    count = 0
                    while True:
                        if time.monotonic() > deadline:
                            raise AcquisitionError('failed_download', 'Download wall timeout')
                        block = response.read(1024**2)
                        if not block:
                            break
                        count += len(block)
                        if count > size:
                            raise AcquisitionError('failed_download', 'Response larger than Content-Length')
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
            if candidate.stat().st_size != size:
                raise AcquisitionError('failed_download', 'Transfer size differs from Content-Length')
            try:
                staged, report = self.verify(candidate, u)
            except AcquisitionError as exc:
                if not exc.retryable:
                    rejected_sha = sha_file(candidate)
                    rejected = self.root / 'rejected' / u['folder'] / (target.stem + '_' + rejected_sha + '.cdf')
                    rejected.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(candidate, rejected)
                    exc.details.update(rejected_path=str(rejected), rejected_sha256=rejected_sha,
                                       transport_length_verified=True, content_length=size, origin=origin)
                raise
            sha = sha_file(candidate)
            previous_sha = sha_file(target) if target.exists() else None
            if previous_sha != sha:
                if previous_sha:
                    history = self.root / 'history' / u['folder']
                    history.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, history / (target.stem + '_' + previous_sha + '.cdf'))
                os.replace(candidate, target)
            self.publish_staged(staged, target)
            return {'status': 'verified', 'unit': u, 'path': str(target.resolve()),
                    'fingerprint': fingerprint(target), 'sha256': sha,
                    'validation_identity': self.validation_identity, 'validation': report,
                    'origin': origin, 'url': url, 'content_length': size,
                    'etag': headers.get('ETag'), 'last_modified': headers.get('Last-Modified'),
                    'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        finally:
            candidate.unlink(missing_ok=True)
            # Failed candidates never leave an expanded scratch cache behind.
            if staged and staged['compressed'] and Path(staged['expanded_path']).exists():
                p = Path(staged['expanded_path'])
                p.unlink(missing_ok=True)
                p.with_suffix('.json').unlink(missing_ok=True)

    def ensure(self, product, date):
        u = unit(product, date)
        key = u['key']
        if key in self.results:
            result = self.results[key]
            if result['status'] == 'verified' and fingerprint(result['path']) != result['fingerprint']:
                raise AcquisitionError('failed_validation', 'Verified input changed during run: ' + key)
            return result
        folder = self.args.input_root / u['folder']
        folder.mkdir(parents=True, exist_ok=True)
        state_path = self.root / 'state' / date.strftime('%Y/%m') / (key + '.json')
        state_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = self.root / 'receipts' / date.strftime('%Y/%m') / (key + '.json')
        with state_path.with_suffix('.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (state_path.exists() or receipt_path.exists()) and not self.args.refresh_downloads:
                try:
                    result = json.loads(state_path.read_text()) if state_path.exists() else {}
                    if result.get('status') != 'verified' and receipt_path.exists():
                        result = json.loads(receipt_path.read_text())
                    if (result['status'] == 'verified' and result['unit'] == u
                            and result['url'].startswith(self.remote_root)
                            and Path(result['path']).resolve().parent in (folder.resolve(), (self.root / 'erg' / u['folder']).resolve())
                            and fingerprint(result['path']) == result['fingerprint']
                            and sha_file(result['path']) == result['sha256']):
                        if Path(result['path']).resolve().parent != folder.resolve():
                            target = folder / Path(result['path']).name
                            if target.exists() and sha_file(target) != result['sha256']:
                                raise ValueError('Shared cache differs from legacy receipt')
                            if not target.exists():
                                fd, tmp = tempfile.mkstemp(prefix=target.name + '.', suffix='.part', dir=folder)
                                try:
                                    with os.fdopen(fd, 'wb') as out, Path(result['path']).open('rb') as inp:
                                        shutil.copyfileobj(inp, out, 1024**2)
                                        out.flush()
                                        os.fsync(out.fileno())
                                    if sha_file(tmp) != result['sha256']:
                                        raise ValueError('Legacy cache changed during migration')
                                    os.replace(tmp, target)
                                finally:
                                    Path(tmp).unlink(missing_ok=True)
                            result.update(path=str(target.resolve()), fingerprint=fingerprint(target))
                        if result['validation_identity'] != self.validation_identity:
                            _, report = self.verify(Path(result['path']), u)
                            result.update(validation_identity=self.validation_identity, validation=report,
                                          revalidated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                        atomic_json(state_path, result)
                        atomic_json(receipt_path, result)
                        self.results[key] = result
                        return result
                except AcquisitionError as exc:
                    if not exc.retryable:
                        result = {'status': exc.status, 'unit': u, 'reason': str(exc),
                                  'attempts': [], 'retryable': False, **exc.details}
                        atomic_json(state_path, result)
                        self.results[key] = result
                        return result
                except (OSError, ValueError, KeyError):
                    pass
            attempts = []
            result = {'status': 'failed_download', 'unit': u, 'reason': 'No verified offline input'}
            if not self.args.offline:
                for number in range(1, self.args.download_attempts + 1):
                    local = None
                    phase = 'catalog'
                    self.last_http = {}
                    try:
                        wait_resources(self.args, self.scratch)
                        catalog = self.catalog(u['folder'])
                        phase = 'file'
                        matches = [(tuple(map(int, m.groups())), name) for name in catalog['names']
                                   if (m := re.fullmatch(re.escape(u['prefix']) + r'_v(\d+)[_.](\d+)\.cdf', name))]
                        if not matches:
                            if any(x.startswith(u['prefix'] + '_v') for x in catalog['names']):
                                raise AcquisitionError('failed_validation', 'Unrecognized version filename')
                            result = {'status': 'remote_absent', 'unit': u, 'evidence': catalog,
                                      'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                            break
                        name = max(matches)[1]
                        target = folder / name
                        # After any failure, prefer new bytes over re-copying the same bad local file.
                        if number == 1 and not self.args.refresh_downloads:
                            local = self.args.input_root / u['folder'] / name
                        url = self.remote_root + u['folder'] + '/' + name
                        result = self.fetch_candidate(url, target, local, u)
                        break
                    except Exception as exc:
                        status = exc.status if isinstance(exc, AcquisitionError) else 'failed_download'
                        if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
                            status = 'failed_authorization'
                        print(f'acquisition {key} attempt {number}: {status}: {exc}', flush=True)
                        attempts.append({'attempt': number, 'status': status, 'reason': str(exc),
                                         'http_status': getattr(exc, 'code', None), 'phase': phase,
                                         **self.last_http, 'http_read_timeout_seconds': self.args.http_timeout,
                                         'download_wall_timeout_seconds': self.args.download_timeout,
                                         **(exc.details if isinstance(exc, AcquisitionError) else {})})
                        result = {'status': status, 'unit': u, 'reason': str(exc),
                                  'retryable': getattr(exc, 'retryable', True),
                                  **(exc.details if isinstance(exc, AcquisitionError) else {})}
                        # Keep a valid index on ordinary file failures; refresh only a stale listed link.
                        if isinstance(exc, urllib.error.HTTPError) and exc.code in (404, 410):
                            self.catalogs.pop(u['folder'], None)
                        if status == 'failed_authorization' or not getattr(exc, 'retryable', True):
                            break
                        if number < self.args.download_attempts:
                            time.sleep(self.args.retry_delay)
            result['attempts'] = attempts
            atomic_json(state_path, result)
            if result['status'] == 'verified':
                atomic_json(receipt_path, result)
            self.results[key] = result
            print(f'acquisition {key}: {result["status"]}', flush=True)
            if result['status'] in ('failed_download', 'failed_authorization') and not self.args.offline:
                self.network_failure_streak += 1
            else:
                self.network_failure_streak = 0
            limit = getattr(self.args, 'max_network_failures', 0)
            if limit and self.network_failure_streak >= limit:
                raise AcquisitionError('failed_download',
                    f'Stopping after {limit} consecutive granule acquisition failures; resolve network then resume',
                    retryable=False, unit=key)
            return result

    def retry_failed(self):
        """Discard only transient failures; retain verified, absent and policy-rejected units."""
        keys = [key for key, result in self.results.items()
                if result['status'] not in ('verified', 'remote_absent')
                and result.get('retryable', True)]
        for key in keys:
            del self.results[key]
        self.network_failure_streak = 0
        return keys

    def prefetch(self, intervals, cadence, progress_mode='auto'):
        """Acquire all distinct granules before calculation; never retain expanded month data."""
        planned = {}
        for day, lo, hi in intervals:
            dates = [day]
            if lo == day:
                dates.insert(0, day - dt.timedelta(days=1))
            if hi == day + dt.timedelta(days=1):
                dates.append(day + dt.timedelta(days=1))
            for date in dates:
                u = unit('lepi', date)
                planned[u['key']] = ('lepi', date)
            step = dt.timedelta(hours=1) if cadence == '64hz' else dt.timedelta(days=1)
            pad = dt.timedelta(seconds=16 if cadence == '64hz' else 32)
            cursor = (lo - pad).replace(minute=0, second=0, microsecond=0)
            if cadence == '8sec':
                cursor = cursor.replace(hour=0)
            while cursor <= hi + pad:
                u = unit(cadence, cursor)
                planned[u['key']] = (cadence, cursor)
                cursor += step
        from tqdm import tqdm
        enabled = progress_mode == 'on' or (progress_mode == 'auto' and sys.stderr.isatty())
        statuses = {}
        with tqdm(total=len(planned), desc='Acquire/verify', unit='file', disable=not enabled,
                  file=sys.stderr, mininterval=1., dynamic_ncols=True) as bar:
            for product, date in sorted(planned.values(), key=lambda x: (x[1], x[0])):
                try:
                    result = self.ensure(product, date)
                    statuses[result['unit']['key']] = result['status']
                    bar.set_postfix(status=result['status'], refresh=False)
                    bar.update(1)
                finally:
                    # Validation expands one CDF at a time. Calculation re-expands per day.
                    cache = self.scratch / 'cdf'
                    if cache.exists():
                        shutil.rmtree(cache)
        return statuses

    def interval(self, day, lo, hi, cadence):
        dates = [day]
        if lo == day:
            dates.insert(0, day - dt.timedelta(days=1))
        if hi == day + dt.timedelta(days=1):
            dates.append(day + dt.timedelta(days=1))
        particles = [self.ensure('lepi', d) for d in dates]
        primary = next(x for x in particles if x['unit']['date'] == day.isoformat())
        mag = []
        if primary['status'] == 'verified':
            step = dt.timedelta(hours=1) if cadence == '64hz' else dt.timedelta(days=1)
            pad = dt.timedelta(seconds=16 if cadence == '64hz' else 32)
            cursor = (lo - pad).replace(minute=0, second=0, microsecond=0)
            if cadence == '8sec':
                cursor = cursor.replace(hour=0)
            while cursor <= hi + pad:
                mag.append(self.ensure(cadence, cursor))
                cursor += step
        results = particles + mag
        failures = [x for x in results if x['status'] not in ('verified', 'remote_absent')]
        # Stable evidence in the signature, not volatile request timestamps/retry histories.
        proof = [{'unit': x['unit']['key'], 'status': x['status'],
                  **({k: x[k] for k in ('fingerprint', 'sha256', 'validation_identity')}
                     if x['status'] == 'verified' else
                     {'absence_evidence': {k: x['evidence'].get(k) for k in
                                           ('url', 'http_status', 'index_sha256')}}
                     if x['status'] == 'remote_absent' else {})} for x in results]
        return ([Path(x['path']) for x in particles if x['status'] == 'verified'],
                [Path(x['path']) for x in mag if x['status'] == 'verified'], proof,
                failures, primary['status'])
