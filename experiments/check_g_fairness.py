"""Fairness checks of baseline G (external assessment of 23 Nov 2026, Sec. 5):
  1. dimension and numerical rank of G's enriched space Z (vs r and n_c);
  2. constant K (block): G online vs FOM-cond (reused factorisation and Delassus matrix);
  3. parameter-dependent K (layered): G online vs FOM-cold and FOM-warm (active set warm-started from the nearest
     training snapshot), the fairest full-order reference when no factorisation can be reused.
Output: results/check_g_fairness/<problem>_seed<seed>.npz and a summary in the log.
Usage:  python experiments/check_g_fairness.py --problem layered --seed 0
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before NumPy (single thread)
    os.environ[_name] = "1"

import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from certcorrect import cli  # noqa: E402
from certcorrect.config import SEEDS, SETTINGS, TAU, TAU_GAP, make_problem, result_path  # noqa: E402
from certcorrect.corrections import EnergyProjection, RightInverse  # noqa: E402
from certcorrect.fom import CondensedSolver, FullOrderSolver, energy_norm  # noqa: E402
from certcorrect.multipliers import MinResidualS1  # noqa: E402
from certcorrect.offline_online import OnlineCertifier  # noqa: E402
from certcorrect.pipeline import EnergyInnerProduct, build_bases, training_snapshots  # noqa: E402
from certcorrect.results_io import save_result  # noqa: E402

LOG = logging.getLogger("check_g_fairness")
EXPERIMENT = "check_g_fairness"
N_TEST = 40
REPEATS = 3
RANK_RTOL = 1e-10


def best_time(func):
    best, out = np.inf, None
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return best, out


def main(args) -> int:
    problem = make_problem(args.problem)
    settings = SETTINGS[args.problem]
    path = result_path(EXPERIMENT, args.problem, args.seed)
    if args.dry_run:
        return cli.EXIT_OK
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + args.seed), settings["n_train"])
    r = max(settings["r_primal"])
    V, W = build_bases(snap, r, settings["r_dual"], settings["dual_kind"])
    inner = EnergyInnerProduct(problem)
    proj = EnergyProjection(problem, V, inner.solve, inner.matrix, RightInverse.from_constraints(problem.B),
                            "compliance", xinv_bt=MinResidualS1(problem.B, inner.solve).xinv_bt)
    online = OnlineCertifier(problem, V, W, inner, projection=proj)
    sv = np.linalg.svd(np.hstack([V, MinResidualS1(problem.B, inner.solve).xinv_bt]), compute_uv=False)
    rank = int(np.sum(sv > RANK_RTOL * sv[0]))
    LOG.info("%s: N %d, n_c %d, r %d | dim Z (after X-orthonormalisation) %d, numerical rank of [V, X^-1 B^T] %d "
             "(r + n_c = %d)", problem.name, problem.n_dof, problem.n_contact, V.shape[1], online.m, rank,
             V.shape[1] + problem.n_contact)
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + args.seed), N_TEST)
    fom = FullOrderSolver(problem)
    cond = None if problem.k_varies else CondensedSolver(problem)
    width = np.where(problem.hi > problem.lo, problem.hi - problem.lo, 1.0)
    t = {k: np.full(N_TEST, np.nan) for k in ("G", "R0s", "fom_cold", "fom_warm", "fom_cond")}
    acc = np.zeros(N_TEST, bool)
    for j, mu in enumerate(mus):
        start = time.perf_counter()
        ref = fom.solve(mu, K=problem.stiffness(mu))
        t["fom_cold"][j] = time.perf_counter() - start
        nearest = int(np.argmin(np.linalg.norm((snap.mus - mu) / width, axis=1)))
        mask = np.zeros(problem.n_contact, bool)
        mask[snap.active[nearest]] = True
        start = time.perf_counter()
        fom.solve(mu, active0=mask, K=problem.stiffness(mu))
        t["fom_warm"][j] = time.perf_counter() - start
        if cond is not None:
            t["fom_cond"][j] = best_time(lambda: cond.solve(mu))[0]
        t["G"][j], res = best_time(lambda: online.evaluate(mu, TAU, TAU_GAP, correction="G"))
        acc[j] = res.accepted
        t["R0s"][j] = best_time(lambda: online.evaluate(mu, TAU, TAU_GAP, correction="R0s"))[0]
    med = {k: float(np.nanmedian(v)) for k, v in t.items()}
    LOG.info("median times: %s", ", ".join(f"{k} {1e3 * v:.2f} ms" for k, v in med.items() if np.isfinite(v)))
    LOG.info("G certified share %.0f %% | G speed-up vs FOM-cold %.1fx, vs FOM-warm %.1fx%s", 100 * acc.mean(),
             med["fom_cold"] / med["G"], med["fom_warm"] / med["G"],
             f", vs FOM-cond {med['fom_cond'] / med['G']:.2f}x" if np.isfinite(med["fom_cond"]) else "")
    save_result(path, {**{f"t_{k}": v for k, v in t.items()}, "accepted_G": acc, "dim_Z": np.array(online.m),
                       "rank_VXB": np.array(rank)},
                {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed, "medians": med})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
