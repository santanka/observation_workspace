#!/usr/bin/env python3
"""Environment-statistics entrypoint with configurable beta_e range parallelism."""

from pathlib import Path

import run_environment_statistics as pipeline
from prepare_electron_beta_statistics_parallel import positive_workers


SERIAL_PARSER = pipeline.build_parser
SERIAL_COMMANDS = pipeline.commands


def build_parser():
    parser = SERIAL_PARSER()
    parser.description = __doc__
    parser.add_argument("--workers", type=positive_workers, default=1,
                        help="Concurrent beta_e range processes; other stages remain serial (default: 1)")
    return parser


def commands(args, root, period):
    steps = SERIAL_COMMANDS(args, root, period)
    for name, command in steps:
        if name == "prepare_beta_e":
            command[1] = str(Path(__file__).resolve().parent / "prepare_electron_beta_statistics_parallel.py")
            command.extend(["--workers", str(args.workers)])
    return steps


def main(argv=None):
    # Adapt only this process. Do not edit modules used by an active legacy job.
    old_parser, old_commands, old_file = pipeline.build_parser, pipeline.commands, pipeline.__file__
    try:
        pipeline.build_parser, pipeline.commands = build_parser, commands
        pipeline.__file__ = __file__
        return pipeline.main(argv)
    finally:
        pipeline.build_parser, pipeline.commands, pipeline.__file__ = old_parser, old_commands, old_file


if __name__ == "__main__":
    raise SystemExit(main())
