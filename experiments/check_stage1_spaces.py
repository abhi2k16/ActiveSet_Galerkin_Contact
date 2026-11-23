"""Checks of the Stage 1 trial spaces (external verification of 23 Nov 2026, Secs. 3 and 6):
  1. numerical rank of every trial space [V, E_k] in the X inner product (requested vs effective dimension);
  2. principal angles (X inner product) between the control (displacement POD, r + k modes) and each enriched space
     of the same requested dimension: cosines near 1 mean the spaces nearly coincide;
  3. the R0 clean-up after the working-set QP: feasibility before clean-up and ||u^ - Z q||_K / ||u^||_K.
Same training data and seeds as experiments/compression_stage1.py; no full-order test solves needed.
Output: results/check_stage1_spaces/layered_seed<seed>.npz.   Usage:  python experiments/check_stage1_spaces.py --seed 0
"""

import logging

import numpy as np

from certcorrect import cli
from certcorrect.bases import pod_basis, svd_dual_basis
from certcorrect.config import SEEDS, result_path
from certcorrect.corrections import RightInverse, repair_r0
from certcorrect.fom import energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.pipeline import EnergyInnerProduct, training_snapshots
from certcorrect.problems import ElasticBlock
from certcorrect.results_io import save_result
from certcorrect.selective import select_training_force, solve_working_set

import compression_stage1 as stage1  # noqa: E402  (same constants and snapshot construction)

LOG = logging.getLogger("check_stage1_spaces")
EXPERIMENT = "check_stage1_spaces"
RANK_RTOL = 1e-10
N_CLEANUP = 10


def x_orthonormal(Z: np.ndarray, X) -> tuple[np.ndarray, int]:
    G = Z.T @ (X @ Z)
    w, Q = np.linalg.eigh(0.5 * (G + G.T))
    keep = w > RANK_RTOL * w.max()
    return Z @ (Q[:, keep] / np.sqrt(w[keep])), int(keep.sum())


def principal_cosines(Q1: np.ndarray, Q2: np.ndarray, X) -> np.ndarray:
    return np.clip(np.linalg.svd(Q1.T @ (X @ Q2), compute_uv=False), 0.0, 1.0)


def main(args) -> int:
    path = result_path(EXPERIMENT, "layered", args.seed)
    if args.dry_run:
        return cli.EXIT_OK
    arrays = {}
    for li, (n_xy, nz) in enumerate(stage1.LEVELS):
        problem = ElasticBlock(n_xy=n_xy, nz=nz, layered=True)
        snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + args.seed), stage1.N_TRAIN)
        inner = EnergyInnerProduct(problem)
        X = inner.matrix
        s1 = MinResidualS1(problem.B, inner.solve)
        Bp = RightInverse.from_constraints(problem.B)
        V = pod_basis(snap.U, stage1.R_PRIMAL)
        Phi = pod_basis(stage1.response_snapshots(problem, snap), max(stage1.KS), K=X)
        Wsvd = svd_dual_basis(snap.L, min(max(stage1.KS), snap.L.shape[1]))
        mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + args.seed), N_CLEANUP)
        LOG.info("n_c %d:", problem.n_contact)
        for k in [k for k in stage1.KS if k <= min(problem.n_contact, stage1.N_TRAIN)]:
            spaces = {"coordinate": np.hstack([V, s1.xinv_bt[:, select_training_force(snap.L, k)]]),
                      "response_pod": np.hstack([V, Phi[:, :k]]),
                      "compressed": np.hstack([V, s1.xinv_bt @ Wsvd[:, :k]]),
                      "control": pod_basis(snap.U, stage1.R_PRIMAL + k)}
            ortho = {name: x_orthonormal(Z, X) for name, Z in spaces.items()}
            Qc = ortho["control"][0]
            parts = []
            for name, (Q, rank) in ortho.items():
                cos = principal_cosines(Q, Qc, X) if name != "control" else np.ones(rank)
                feas, clean = [], []
                for mu in mus:
                    K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
                    Z = spaces[name]
                    q, ok, _ = solve_working_set(Z.T @ (K @ Z), Z.T @ f, np.asarray(problem.B @ Z), c)
                    v = Z @ q
                    u = repair_r0(v, problem.B, c, Bp)
                    feas.append(ok)
                    clean.append(energy_norm(K, u - v) / max(energy_norm(K, u), 1e-300))
                arrays[f"L{li}|k{k}|{name}|cos"] = cos
                arrays[f"L{li}|k{k}|{name}|clean"] = np.array(clean)
                parts.append(f"{name}: rank {rank}/{Z.shape[1]}, cos min {cos.min():.3f} mean {np.mean(cos ** 2):.3f}, "
                             f"QP feasible {100 * np.mean(feas):.0f} %, clean-up max {max(clean):.1e}")
            LOG.info("  k=%2d | %s", k, " | ".join(parts))
    save_result(path, arrays, {"experiment": EXPERIMENT, "seed": args.seed, "rank_rtol": RANK_RTOL})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
