"""One isolated acquisition unit; SIGALRM remains on its main thread."""
import argparse
import datetime as dt
import fcntl
import json
import signal
from pathlib import Path
from types import SimpleNamespace

from acquisition import Acquisition, AcquisitionError
from batch import atomic_json


class SerialValidationAcquisition(Acquisition):
    """Downloads overlap; expanded CDFs and validators use one shared slot."""
    validation_lock = None

    def verify(self, path, u):
        if self.validation_lock is None:
            self.validation_lock = Path(self.args.validation_lock).open('a+')
            try:
                fcntl.flock(self.validation_lock, fcntl.LOCK_EX)
            except BaseException:
                self.release_validation()
                raise
        try:
            return super().verify(path, u)
        except BaseException:
            self.release_validation()
            raise

    def release_validation(self):
        if self.validation_lock is not None:
            self.validation_lock.close()
            self.validation_lock = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    options = parser.parse_args()
    payload = json.loads(options.request.read_text())
    args = SimpleNamespace(**payload['args'])
    for key in ('input_root', 'output_root', 'scratch_root'):
        if hasattr(args, key):
            setattr(args, key, Path(getattr(args, key)))
    args.max_network_failures = 0  # The parent owns the global circuit breaker.
    scratch = Path(payload['scratch'])
    scratch.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Acquisition stop signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    acquisition = SerialValidationAcquisition(args, scratch)
    acquisition.catalogs.update(payload['catalogs'])
    try:
        try:
            result = acquisition.ensure(payload['product'], dt.datetime.fromisoformat(payload['date']))
            atomic_json(options.result, {'result': result, 'catalogs': acquisition.catalogs})
        except AcquisitionError as exc:
            atomic_json(options.result, {'error': {'status': exc.status, 'reason': str(exc),
                                                  'retryable': exc.retryable, 'details': exc.details}})
    finally:
        # Clean before releasing the shared slot: at most one expanded unit survives.
        import shutil
        shutil.rmtree(scratch / 'cdf', ignore_errors=True)
        acquisition.release_validation()


if __name__ == '__main__':
    main()
