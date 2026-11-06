"""Real ATT layout regressions plus strict rejection of ambiguous numeric text."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import convert
import attitude_support as a


class AttitudeTests(unittest.TestCase):
    normal = '2020-01-29/11:09:10.3391 7.5019 135.2657 26.2539 0.0093 170.9215 114.7291 I G 0.3736 43.5690 3.4354 135.2657 26.2539 -151515.3574 0.0'

    def test_joined_signed_column_keeps_axes_and_quality(self):
        joined = self.normal.replace('26.2539 -151515', '26.2539-151515')
        t, angles, quality, count = a.parse_attitude('# J2000\n' + joined)
        self.assertEqual(t, ['2020-01-29/11:09:10.3391'])
        np.testing.assert_array_equal(angles, [[135.2657, 26.2539]])
        np.testing.assert_array_equal(quality, [['I', 'G', '-151515.3574', '0.0']])
        self.assertEqual(count, 1)
        omitted = joined.replace(' I G ', ' ')
        _, axes2, flags2, count = a.parse_attitude('# J2000\n' + omitted)
        np.testing.assert_array_equal(axes2, angles)
        np.testing.assert_array_equal(flags2, [['', '', '-151515.3574', '0.0']])
        self.assertEqual(count, 1)

    def test_normal_14_and_16_columns_match_legacy(self):
        for line in (self.normal, self.normal.replace(' I G ', ' ')):
            content = '# J2000\n' + line
            t, angles, quality, count = a.parse_attitude(content)
            old_t, old_quality = convert.attitude_quality_rows(content)
            self.assertEqual(t, old_t)
            np.testing.assert_array_equal(quality, old_quality)
            self.assertEqual(count, 0)
            np.testing.assert_array_equal(angles, [[135.2657, 26.2539]])

    def test_exponents_nan_and_reject_junk(self):
        self.assertEqual(a.numeric_tokens('1.2e-3-4.5E+2'), ['1.2e-3', '-4.5E+2'])
        self.assertEqual(a.numeric_tokens('NaN-4.5'), ['NaN', '-4.5'])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            a.numeric_tokens('1.2.3')
        for line in (self.normal + ' garbage', self.normal.replace('135.2657', 'oops', 1)):
            with self.assertRaises(ValueError):
                a.parse_attitude('# J2000\n' + line)
        with self.assertRaisesRegex(ValueError, 'J2000'):
            a.parse_attitude(self.normal)

    def test_unrelated_support_failure_not_hidden(self):
        support = a.Support(SimpleNamespace())
        with patch.object(convert.Support, 'load', side_effect=RuntimeError('SHA256 mismatch')):
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                support.load(None)


if __name__ == '__main__':
    unittest.main()
