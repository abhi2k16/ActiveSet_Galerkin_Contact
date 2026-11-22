"""Controlled lift-off sequence on the full-order membrane: a direct check of paper Proposition 3.3 / Corollary 3.5.

For a converged full-order solution (u, lam), let phi = K^-1 B^T lam / ||.||_K (the compliance response to the true
contact force; phi >= 0 because K is an M-matrix for the membrane, and phi > 0 on the strictly active rows). The
predictions u~_eps = u + eps phi lift every strictly active row off the obstacle by eps phi_i with the force lam_i
unchanged, i.e. a = 0 in Corollary 3.5:
  * R0 leaves them open        -> missed rows M = A_+, certificate ~ eps^(1/2) (half rate);
  * R0s with A^ = A_+ snaps them -> M = empty,      certificate ~ eps       (full rate, effectivity <= 2/sqrt3).
Both use S1 with X = K. Output: results/liftoff/<problem>_seed<seed>.npz and fitted rates in the log.

Usage:  python experiments/liftoff_sequence.py --problem membrane --seed 0
"""

import logging

import numpy as np
from scipy.sparse.linalg import factorized

from certcorrect import cli
from certcorrect.analysis import loglog_slope
from certcorrect.certificate import certify
from certcorrect.config import SEEDS, make_problem, result_path
from certcorrect.corrections import RightInverse, repair_r0, repair_r0_snap
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.results_io import save_result

LOG = logging.getLogger("liftoff_sequence")
EXPERIMENT = "liftoff"
EPSILONS = np.geomspace(1e-1, 1e-6, 11)     # lift-off amplitude relative to ||u||_K
N_QUERIES = 10
NONNEG_TOL = 1e-12


def run(problem, seed: int) -> dict[str, np.ndarray]:
    if problem.k_varies:
        raise ValueError("the lift-off sequence is defined for a constant K (X = K)")
    K = problem.stiffness(problem.lo)
    solve_k = factorized(K.tocsc())
    s1, Bp, B = MinResidualS1(problem.B, solve_k), RightInverse.from_constraints(problem.B), problem.B
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), N_QUERIES)
    shape = (N_QUERIES, len(EPSILONS))
    out = {f"{k}_{m}": np.full(shape, np.nan) for k in ("error", "delta") for m in ("R0", "R0s")}
    out["eps"] = EPSILONS
    for i, mu in enumerate(mus):
        f, c = problem.load(mu), problem.offset(mu)
        ref = FullOrderSolver(problem).solve(mu, K=K)
        if not ref.converged or ref.strictly_active.size == 0:
            LOG.warning("query %d skipped (no converged solution with contact)", i)
            continue
        phi = solve_k(B.T @ ref.lam)
        if phi.min() < -NONNEG_TOL * np.abs(phi).max():
            raise ValueError("phi has negative entries: K is not an M-matrix for this problem")
        phi *= energy_norm(K, ref.u) / energy_norm(K, phi)
        for j, eps in enumerate(EPSILONS):
            u_tilde = ref.u + eps * phi
            for name, u_hat in (("R0", repair_r0(u_tilde, B, c, Bp)),
                                ("R0s", repair_r0_snap(u_tilde, B, c, Bp, ref.strictly_active))):
                g_hat = np.maximum(B @ u_hat - c, 0.0)
                lam, _ = s1.solve(f - K @ u_hat, g_hat, 1.0)
                out[f"error_{name}"][i, j] = energy_norm(K, ref.u - u_hat)
                out[f"delta_{name}"][i, j] = certify(K, f, B, c, u_hat, lam, solve_k, 1.0).delta
    return out


def main(args) -> int:
    problem = make_problem(args.problem)
    path = result_path(EXPERIMENT, args.problem, args.seed)
    LOG.info("%s: %d queries x %d amplitudes -> %s", problem.name, N_QUERIES, len(EPSILONS), path)
    if args.dry_run:
        return cli.EXIT_OK
    out = run(problem, args.seed)
    for name in ("R0", "R0s"):
        e, d = out[f"error_{name}"], out[f"delta_{name}"]
        small = EPSILONS <= 1e-3                                    # asymptotic part of the sequence
        LOG.info("%-3s rate of Delta in ||e||: all %.2f, eps <= 1e-3: %.2f | effectivity at the smallest eps: "
                 "median %.3g", name, loglog_slope(e.ravel(), d.ravel()),
                 loglog_slope(e[:, small].ravel(), d[:, small].ravel()), np.nanmedian(d[:, -1] / e[:, -1]))
    save_result(path, out, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed,
                            "epsilons": EPSILONS.tolist()})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
