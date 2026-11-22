"""X4 multiplier variants S0, S1, S1-restricted, S2: sharpness and time.

Criterion (fixed, plan Sec. 6): S1-restricted within a factor 1.5 of S1 at >= 5x lower cost, or reported otherwise.

Output: ``results/x4/<problem>_seed<seed>.npz``. Status: weeks 5-6.

Usage:  python experiments/x4_multipliers.py --problem membrane --seed 0
"""

from certcorrect import cli


def main(args) -> int:
    if args.dry_run:
        return cli.EXIT_OK
    raise NotImplementedError("weeks 5-6")


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
