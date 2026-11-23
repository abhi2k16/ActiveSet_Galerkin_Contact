"""Stage 1 of CONTACT_RESPONSE_PLAN.md: compression feasibility of the contact-response space (accuracy only).

Layered block, n_c = 225, 361, 529; r = 15; 60 training snapshots; 30 test queries. Constrained Galerkin solution
(all n_c constraints; working-set QP; R0 clean-up) on [V, E_k] for extra spaces of equal dimension k:
  coordinate     X^-1 B_S^T, S = top-k training-force rows (frequency of strict activity, ties by mean force);
  response_pod   k X-orthonormal POD modes of the contact-response snapshots u_j - u_f(mu_j) = C(mu_j) lam_j;
  compressed     X^-1 B^T W_k, W_k = k leading left singular vectors of the multiplier snapshots;
  control        displacement POD with r + k modes and no compliance columns.
Certificate: S1 with X = K(mu_bar), evaluated at full order; tau = 0.05.

Stop criterion (fixed in the plan before running): compression (response_pod or compressed) is pursued only if, at
n_c = 529, it reaches the certified share of coordinate selection at k = 0.2 n_c (or 95 %, whichever is lower) with at
most half that k. The control "matches" (line stops) if, at n_c = 529 and the reference k, it certifies at least
the share of coordinate selection AND its median error is not larger (operational form fixed before the full run).

Output: results/compression_stage1/layered_seed<seed>.npz.   Usage:  python experiments/compression_stage1.py --seed 0
"""

import logging

import numpy as np
from scipy.sparse.linalg import factorized

from certcorrect import cli
from certcorrect.bases import pod_basis, svd_dual_basis
from certcorrect.certificate import certify, is_accepted
from certcorrect.config import SEEDS, TAU, result_path
from certcorrect.corrections import RightInverse, repair_r0
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.pipeline import EnergyInnerProduct, training_snapshots
from certcorrect.problems import ElasticBlock
from certcorrect.results_io import save_result
from certcorrect.selective import select_training_force, solve_working_set

LOG = logging.getLogger("compression_stage1")
EXPERIMENT = "compression_stage1"
LEVELS = ((16, 4), (20, 5), (24, 6))
N_TRAIN, N_TEST, R_PRIMAL = 60, 30, 15
KS = (5, 10, 20, 40, 80)
SPACES = ("coordinate", "response_pod", "compressed", "control")
DECAY_AT = (1, 5, 10, 20, 40, 80)


def response_snapshots(problem, snap) -> np.ndarray:
    """``u_j - K(mu_j)^-1 f(mu_j) = C(mu_j) lam_j`` for the training snapshots."""
    cols = []
    for j, mu in enumerate(snap.mus):
        u_f = factorized(problem.stiffness(mu).tocsc())(problem.load(mu))
        cols.append(snap.U[:, j] - u_f)
    return np.column_stack(cols)


def x_decay(R: np.ndarray, X) -> np.ndarray:
    """Singular values of the snapshot matrix in the X-norm, normalised by the largest."""
    G = R.T @ (X @ R)
    s = np.sqrt(np.clip(np.linalg.eigvalsh(0.5 * (G + G.T))[::-1], 0.0, None))
    return s / s[0]


def galerkin(problem, Z: np.ndarray, mu, Bp) -> tuple[np.ndarray, bool]:
    K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
    H = Z.T @ (K @ Z)
    q, feasible, _ = solve_working_set(H, Z.T @ f, np.asarray(problem.B @ Z), c)
    return repair_r0(Z @ q, problem.B, c, Bp), feasible


def run_level(n_xy: int, nz: int, seed: int) -> dict:
    problem = ElasticBlock(n_xy=n_xy, nz=nz, layered=True)
    n_c = problem.n_contact
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), N_TRAIN)
    inner = EnergyInnerProduct(problem)
    Bp = RightInverse.from_constraints(problem.B)
    s1 = MinResidualS1(problem.B, inner.solve)
    V = pod_basis(snap.U, R_PRIMAL)
    R = response_snapshots(problem, snap)
    Phi = pod_basis(R, max(KS), K=inner.matrix)
    Wsvd = svd_dual_basis(snap.L, min(max(KS), snap.L.shape[1]))
    decay_resp = x_decay(R, inner.matrix)
    decay_lam = np.linalg.svd(snap.L, compute_uv=False)
    decay_lam = decay_lam / decay_lam[0]
    ks = [k for k in KS if k <= min(n_c, N_TRAIN)]
    spaces = {}
    for k in ks:
        spaces[("coordinate", k)] = np.hstack([V, s1.xinv_bt[:, select_training_force(snap.L, k)]])
        if k <= Phi.shape[1]:
            spaces[("response_pod", k)] = np.hstack([V, Phi[:, :k]])
        if k <= Wsvd.shape[1]:
            spaces[("compressed", k)] = np.hstack([V, s1.xinv_bt @ Wsvd[:, :k]])
        spaces[("control", k)] = pod_basis(snap.U, R_PRIMAL + k)
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), N_TEST)
    fom = FullOrderSolver(problem)
    keys = list(spaces)
    err = np.full((len(keys), N_TEST), np.nan)
    acc = np.zeros((len(keys), N_TEST), bool)
    feas = np.ones((len(keys), N_TEST), bool)
    dims = np.array([spaces[key].shape[1] for key in keys])
    for j, mu in enumerate(mus):
        K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
        ref = fom.solve(mu, K=K)
        if not ref.converged:
            continue
        norm_u, alpha = energy_norm(K, ref.u), inner.alpha(mu)
        for i, key in enumerate(keys):
            u_hat, feas[i, j] = galerkin(problem, spaces[key], mu, Bp)
            lam, _ = s1.solve(f - K @ u_hat, np.maximum(problem.B @ u_hat - c, 0.0), alpha)
            cert = certify(K, f, problem.B, c, u_hat, lam, inner.solve, alpha)
            err[i, j] = energy_norm(K, ref.u - u_hat) / norm_u
            acc[i, j] = is_accepted(cert.delta, energy_norm(K, u_hat), TAU)
    return {"n_contact": n_c, "keys": keys, "dims": dims, "err": err, "acc": acc, "feas": feas,
            "decay_resp": decay_resp, "decay_lam": decay_lam}


def evaluate_stop(lv: dict) -> tuple[bool, str]:
    """The pre-registered criterion at the largest level."""
    n_c = lv["n_contact"]
    share = {key: float(np.nanmean(lv["acc"][i])) for i, key in enumerate(lv["keys"])}
    coord_ks = [k for (s, k) in share if s == "coordinate"]
    k_ref = max(k for k in coord_ks if k <= 0.2 * n_c) if any(k <= 0.2 * n_c for k in coord_ks) else max(coord_ks)
    target = min(share[("coordinate", k_ref)], 0.95)
    hits = [(s, k) for (s, k), v in share.items() if s in ("response_pod", "compressed") and k <= k_ref / 2
            and v >= target]
    med = {key: float(np.nanmedian(lv["err"][i])) for i, key in enumerate(lv["keys"])}
    control_matches = ("control", k_ref) in share and share[("control", k_ref)] >= share[("coordinate", k_ref)] \
        and med[("control", k_ref)] <= med[("coordinate", k_ref)]
    msg = (f"reference: coordinate k={k_ref} certifies {100 * share[('coordinate', k_ref)]:.0f} %, target "
           f"{100 * target:.0f} % with k <= {k_ref // 2}; compression hits: {hits or 'none'}; control matches "
           f"coordinate at k_ref (share and error): {control_matches}")
    return bool(hits) and not control_matches, msg


def main(args) -> int:
    path = result_path(EXPERIMENT, "layered", args.seed)
    LOG.info("levels %s, k %s, spaces %s -> %s", LEVELS, KS, SPACES, path)
    if args.dry_run:
        return cli.EXIT_OK
    arrays, meta = {}, []
    for li, (n_xy, nz) in enumerate(LEVELS):
        lv = run_level(n_xy, nz, args.seed)
        LOG.info("n_c %d: response-snapshot X-singular values (normalised) at %s: %s | multiplier snapshots: %s",
                 lv["n_contact"], DECAY_AT,
                 np.array2string(lv["decay_resp"][[min(i - 1, len(lv["decay_resp"]) - 1) for i in DECAY_AT]],
                                 precision=2),
                 np.array2string(lv["decay_lam"][[min(i - 1, len(lv["decay_lam"]) - 1) for i in DECAY_AT]],
                                 precision=2))
        for i, (space, k) in enumerate(lv["keys"]):
            ok = np.isfinite(lv["err"][i])
            fails = int(np.sum(lv["acc"][i][ok] & (lv["err"][i][ok] > TAU)))
            LOG.info("  %-13s k=%3d dim %3d | err %.3f %% | cert %5.1f %% (fails %d) | QP feasible %3.0f %%", space, k,
                     lv["dims"][i], 100 * np.nanmedian(lv["err"][i]), 100 * np.mean(lv["acc"][i][ok]), fails,
                     100 * np.mean(lv["feas"][i][ok]))
        for key in ("err", "acc", "feas", "dims", "decay_resp", "decay_lam"):
            arrays[f"L{li}|{key}"] = np.asarray(lv[key])
        meta.append({"n_contact": int(lv["n_contact"]), "keys": [list(k) for k in lv["keys"]]})
    pursue, msg = evaluate_stop(lv)
    LOG.info("STOP CRITERION (n_c = %d): compression %s — %s", lv["n_contact"], "PURSUED" if pursue else "NOT pursued",
             msg)
    save_result(path, arrays, {"experiment": EXPERIMENT, "seed": args.seed, "levels": meta, "pursue": pursue,
                               "criterion": msg})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
