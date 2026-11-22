"""X3 active-set prediction: recall of A_+ and over-snap share for the three A^ rules.

Criterion (fixed, plan Sec. 6): report the trade-off; choose the rule by certified share.

Output: ``results/x3/<problem>_seed<seed>.npz``. Status: weeks 5-6.

Usage:  python experiments/x3_active_set.py --problem membrane --seed 0
"""

from certcorrect import cli


def main(args) -> int:
    if args.dry_run:
        return cli.EXIT_OK
    raise NotImplementedError("weeks 5-6")


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
