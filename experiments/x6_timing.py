"""X6 full pipeline timing and break-even against FOM-cold and FOM-cond.

Criterion (fixed, plan Sec. 6): net speedup and certified share reported together.

Output: ``results/x6/<problem>_seed<seed>.npz``. Status: weeks 7-8.

Usage:  python experiments/x6_timing.py --problem membrane --seed 0
"""

from certcorrect import cli


def main(args) -> int:
    if args.dry_run:
        return cli.EXIT_OK
    raise NotImplementedError("weeks 7-8")


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
