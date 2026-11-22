"""X7 comparison with A, G and CS at equal offline budget.

Criterion (fixed, plan Sec. 6): effectivity, share certified, offline FOM solves and online cost reported.

Output: ``results/x7/<problem>_seed<seed>.npz``. Status: weeks 9-10.

Usage:  python experiments/x7_baselines.py --problem membrane --seed 0
"""

from certcorrect import cli


def main(args) -> int:
    if args.dry_run:
        return cli.EXIT_OK
    raise NotImplementedError("weeks 9-10")


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
