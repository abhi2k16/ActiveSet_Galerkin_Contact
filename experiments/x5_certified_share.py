"""X5 certification share at tau = 0.05: in range, load extrapolation, patch shift.

Criterion (fixed, plan Sec. 6): zero accepted silent failures on the deterministic path; share reported per method.

Output: ``results/x5/<problem>_seed<seed>.npz``. Status: weeks 11-12.

Usage:  python experiments/x5_certified_share.py --problem membrane --seed 0
"""

from certcorrect import cli


def main(args) -> int:
    if args.dry_run:
        return cli.EXIT_OK
    raise NotImplementedError("weeks 11-12")


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
