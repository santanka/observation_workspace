"""Inclusive YYYYMM period arguments shared by statistical plotting commands."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class StatisticsPeriod:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive

    @property
    def last_day(self):
        return self.end - pd.Timedelta(1, unit="ns")

    @property
    def key(self):
        return f"{self.start:%Y%m}-{self.last_day:%Y%m}"

    @property
    def date_suffix(self):
        return f"{self.start:%Y%m%d}_{self.last_day:%Y%m%d}"


def add_period_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int, help="One calendar year (compatibility option).")
    group.add_argument("--start-month", help="First month, YYYYMM, inclusive.")
    parser.add_argument("--end-month", help="Last month, YYYYMM, inclusive; required with --start-month.")


def resolve_period(args):
    if args.year is not None:
        if args.end_month is not None:
            raise ValueError("--end-month cannot be combined with --year")
        return StatisticsPeriod(pd.Timestamp(year=args.year, month=1, day=1),
                                pd.Timestamp(year=args.year + 1, month=1, day=1))
    if args.end_month is None:
        raise ValueError("--end-month is required with --start-month")
    def month(value):
        if len(value) != 6 or not value.isdecimal():
            raise ValueError(f"Expected YYYYMM: {value}")
        return pd.Timestamp(datetime.strptime(value, "%Y%m"))
    start, last = month(args.start_month), month(args.end_month)
    if start > last:
        raise ValueError("--start-month must not exceed --end-month")
    return StatisticsPeriod(start, last + pd.offsets.MonthBegin(1))
