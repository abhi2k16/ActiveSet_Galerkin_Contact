"""Shared command-line setup for experiment scripts: common arguments, logging and exit codes."""

import argparse
import logging
import sys
from collections.abc import Callable

from .config import PROBLEMS

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_BAD_INPUT = 2
EXIT_NOT_IMPLEMENTED = 3
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--problem", choices=sorted(PROBLEMS), default="membrane")
    parser.add_argument("--seed", type=int, default=0, help="replicate seed (offsets the seeds in config.SEEDS)")
    parser.add_argument("--dry-run", action="store_true", help="validate arguments and report the plan; write nothing")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level), format=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


def run(main: Callable[[argparse.Namespace], int], parser: argparse.ArgumentParser) -> None:
    """Parse arguments, run ``main`` and exit with a meaningful code (errors are logged, never swallowed)."""
    args = parser.parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger(parser.prog)
    try:
        code = main(args)
    except NotImplementedError as exc:
        log.error("not implemented yet: %s", exc)
        code = EXIT_NOT_IMPLEMENTED
    except (ValueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        code = EXIT_BAD_INPUT
    sys.exit(code)
