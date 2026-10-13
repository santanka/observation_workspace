"""Terminal progress for attempted five-minute intervals; independent of scientific outputs."""
from contextlib import redirect_stdout
import sys


class Progress:
    def __init__(self, total, mode='auto'):
        self.total = total
        self.enabled = mode == 'on' or (mode == 'auto' and sys.stderr.isatty())
        self.bar = self.redirect = None

    def __enter__(self):
        if self.enabled:
            from tqdm import tqdm
            from tqdm.contrib import DummyTqdmFile
            self.bar = tqdm(total=self.total, desc='LEP-i', unit='chunk', file=sys.stderr,
                            dynamic_ncols=True, mininterval=1., smoothing=0.)
            # Preserve stdout logs/JSON, clearing and restoring the terminal bar around messages.
            self.redirect = redirect_stdout(DummyTqdmFile(sys.stdout))
            self.redirect.__enter__()
        return self

    def current(self, time, phase):
        if self.bar is not None:
            self.bar.set_description_str(f'LEP-i {time:%Y-%m-%d %H:%M} UTC {phase}', refresh=False)

    def restart(self, total):
        self.total = total
        if self.bar is not None:
            self.bar.reset(total=total)
            self.bar.set_description_str('LEP-i acquisition retry', refresh=False)

    def finish(self, counts):
        if self.bar is not None:
            empty = counts['no_data'] + counts['remote_absent']
            failed = sum(n for status, n in counts.items()
                         if status not in {'complete', 'skipped_verified', 'no_data', 'remote_absent'})
            self.bar.set_postfix(ok=counts['complete'], skip=counts['skipped_verified'],
                                 empty=empty, failed=failed, refresh=False)
            self.bar.update(1)

    def __exit__(self, *exc):
        try:
            if self.redirect is not None:
                self.redirect.__exit__(*exc)
        finally:
            if self.bar is not None:
                self.bar.close()
