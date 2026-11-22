"""X8 Level 2 finite-precision verification.

Criterion (fixed, plan Sec. 6): verified and unverified certificates differ by < 1 % on all tested queries, or the difference is reported.

Output: ``results/x8/<problem>_seed<seed>.npz``. Status: week 13.

Usage:  python experiments/x8_finite_precision.py --problem membrane --seed 0
"""

from certcorrect import cli


def main(args) -> int:
    if args.dry_run:
        return cli.EXIT_OK
    raise NotImplementedError("week 13")


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
