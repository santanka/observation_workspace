import argparse
import unittest

from statistics_period import add_period_arguments, resolve_period


class PeriodTests(unittest.TestCase):
    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        add_period_arguments(parser)
        return resolve_period(parser.parse_args(argv))

    def test_three_years_includes_last_december(self):
        period = self.parse("--start-month", "202001", "--end-month", "202212")
        self.assertEqual(period.key, "202001-202212")
        self.assertEqual(str(period.end.date()), "2023-01-01")
        self.assertEqual(period.date_suffix, "20200101_20221231")

    def test_year_compatibility_and_single_month(self):
        self.assertEqual(self.parse("--year", "2022").key, "202201-202212")
        self.assertEqual(self.parse("--start-month", "202002", "--end-month", "202002").date_suffix,
                         "20200201_20200229")

    def test_missing_reversed_and_malformed_months_stop(self):
        for argv in [("--start-month", "202001"),
                     ("--start-month", "202212", "--end-month", "202001"),
                     ("--start-month", "2020-01", "--end-month", "202212"),
                     ("--year", "2022", "--end-month", "202212")]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.parse(*argv)
