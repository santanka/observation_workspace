"""Run scientific worker checks in isolation from unittest's resource limits."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class RecoveryTests(unittest.TestCase):
    def test_known_truth_recovery(self):
        env = {**os.environ, 'SPACEPY': '/tmp/lepi_recovery_spacepy', 'MPLCONFIGDIR': '/tmp/lepi_recovery_mpl'}
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('check_recovery_worker.py'))],
                                env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:]+result.stderr[-5000:])


if __name__ == '__main__':
    unittest.main()
