"""Scaling of the online cost with the number of contact rows n_c (plan D12, option 3): R0s vs Bs vs baseline G.

Layered block refined in LEVELS (n_xy, nz); fixed basis sizes, so only the mesh and n_c change. Per level: offline
stage (training snapshots, bases, representers incl. the enriched space Z), then per test query the online evaluation
(certcorrect.offline_online) of
  R0s  ROM + right-inverse snap repair + S1 + certificate          (no dense (r + n_c) solve)
  Bs   ROM + energy projection with snap in Z = [V, X^-1 B^T] + S1 + certificate
  G    Galerkin contact solve in Z (no ROM) + S1 + certificate     (baseline, Fauque-type)
with the gap rule (config.TAU_GAP), tau = config.TAU, X = K(mu_bar); FOM-cold time; true error offline.
Decision question (D12): is there a range of n_c in which R0s gives a larger net speed-up than G, or in which G's
online cost becomes prohibitive?

Output: results/scaling_nc/layered_seed<seed>.npz.   Usage:  python experiments/scaling_nc.py --seed 0
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before NumPy (single thread)
    os.environ[_name] = "1"

import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from certcorrect import cli  # noqa: E402
from certcorrect.analysis import loglog_slope  # noqa: E402
from certcorrect.config import SEEDS, TAU, TAU_GAP, result_path  # noqa: E402
from certcorrect.corrections import EnergyProjection, RightInverse  # noqa: E402
from certcorrect.fom import FullOrderSolver, energy_norm  # noqa: E402
from certcorrect.multipliers import MinResidualS1  # noqa: E402
from certcorrect.offline_online import STAGES, OnlineCertifier  # noqa: E402
from certcorrect.pipeline import EnergyInnerProduct, build_bases, training_snapshots  # noqa: E402
from certcorrect.problems import ElasticBlock  # noqa: E402
from certcorrect.results_io import save_result  # noqa: E402

LOG = logging.getLogger("scaling_nc")
EXPERIMENT = "scaling_nc"
LEVELS = ((8, 2), (12, 3), (16, 4), (20, 5), (24, 6))     # (n_xy, nz): n_c = (n_xy - 1)^2
N_TRAIN, N_TEST, R_PRIMAL, R_DUAL = 60, 30, 15, 12
METHODS = ("R0s", "Bs", "G")
REPEATS = 3


def best_time(func):
    best, out = np.inf, None
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return best, out


def run_level(n_xy: int, nz: int, seed: int) -> dict:
    problem = ElasticBlock(n_xy=n_xy, nz=nz, layered=True)
    t0 = time.perf_counter()
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), N_TRAIN)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    V, W = build_bases(snap, R_PRIMAL, R_DUAL, "nmf")
    inner = EnergyInnerProduct(problem)
    Bp = RightInverse.from_constraints(problem.B)
    proj = EnergyProjection(problem, V, inner.solve, inner.matrix, Bp, "compliance",
                            xinv_bt=MinResidualS1(problem.B, inner.solve).xinv_bt)
    online = OnlineCertifier(problem, V, W, inner, Bp, projection=proj)
    t_offline = time.perf_counter() - t0
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), N_TEST)
    fom = FullOrderSolver(problem)
    out = {"n_dof": problem.n_dof, "n_contact": problem.n_contact, "m": online.m, "t_train": t_train,
           "t_offline": t_offline, "t_fom": np.full(N_TEST, np.nan), "norm_u": np.full(N_TEST, np.nan)}
    for m in METHODS:
        out[f"t_{m}"] = np.full(N_TEST, np.nan)
        out[f"accepted_{m}"] = np.zeros(N_TEST, bool)
        out[f"error_{m}"] = np.full(N_TEST, np.nan)
        for s in STAGES:
            out[f"t_{s}_{m}"] = np.full(N_TEST, np.nan)
    for j, mu in enumerate(mus):
        start = time.perf_counter()
        ref = fom.solve(mu, K=problem.stiffness(mu))
        out["t_fom"][j] = time.perf_counter() - start
        if not ref.converged:
            continue
        K = problem.stiffness(mu)
        out["norm_u"][j] = energy_norm(K, ref.u)
        for m in METHODS:
            t, res = best_time(lambda: online.evaluate(mu, TAU, TAU_GAP, correction=m))
            out[f"t_{m}"][j], out[f"accepted_{m}"][j] = t, res.accepted
            out[f"error_{m}"][j] = energy_norm(K, ref.u - online.displacement(res))
            for s in STAGES:
                out[f"t_{s}_{m}"][j] = res.seconds[s]
    return out


def level_summary(lv: dict) -> dict:
    ok = np.isfinite(lv["norm_u"])
    t_fom = float(np.nanmedian(lv["t_fom"]))
    rows = {}
    for m in METHODS:
        share = float(np.mean(lv[f"accepted_{m}"][ok]))
        t_on = float(np.nanmedian(lv[f"t_{m}"]))
        rows[m] = {"share": share, "t_online": t_on, "speedup": t_fom / (t_on + (1 - share) * t_fom),
                   "error": float(np.nanmedian(lv[f"error_{m}"][ok] / lv["norm_u"][ok])),
                   "stages": {s: float(np.nanmedian(lv[f"t_{s}_{m}"])) for s in STAGES}}
    return {"t_fom": t_fom, "methods": rows}


def main(args) -> int:
    path = result_path(EXPERIMENT, "layered", args.seed)
    LOG.info("levels %s, n_train %d, n_test %d, r %d, r_dual %d -> %s", LEVELS, N_TRAIN, N_TEST, R_PRIMAL, R_DUAL, path)
    if args.dry_run:
        return cli.EXIT_OK
    arrays, summaries = {}, []
    for k, (n_xy, nz) in enumerate(LEVELS):
        lv = run_level(n_xy, nz, args.seed)
        sm = level_summary(lv)
        summaries.append({"n_dof": int(lv["n_dof"]), "n_contact": int(lv["n_contact"]), "m": int(lv["m"]),
                          "t_offline": float(lv["t_offline"]), "t_train": float(lv["t_train"]), **sm})
        for key, val in lv.items():
            arrays[f"L{k}_{key}"] = np.asarray(val)
        LOG.info("n_dof %6d n_c %4d m %4d | FOM %.0f ms | offline %.1f s (+ training %.1f s) | %s", lv["n_dof"],
                 lv["n_contact"], lv["m"], 1e3 * sm["t_fom"], lv["t_offline"], lv["t_train"], " | ".join(
                     f"{m}: {1e3 * r['t_online']:.1f} ms, cert {100 * r['share']:.0f} %, err {100 * r['error']:.2f} %, "
                     f"net {r['speedup']:.1f}x" for m, r in sm["methods"].items()))
    n_c = np.array([s["n_contact"] for s in summaries])
    for m in METHODS:
        t = np.array([s["methods"][m]["t_online"] for s in summaries])
        LOG.info("%-3s online time ~ n_c^%.2f; correction stage ~ n_c^%.2f", m, loglog_slope(n_c, t),
                 loglog_slope(n_c, np.array([s["methods"][m]["stages"]["correction"] for s in summaries])))
    LOG.info("FOM-cold ~ n_c^%.2f", loglog_slope(n_c, np.array([s["t_fom"] for s in summaries])))
    save_result(path, arrays, {"experiment": EXPERIMENT, "seed": args.seed, "levels": LEVELS, "tau": TAU,
                               "tau_gap": TAU_GAP, "summaries": summaries, "n_train": N_TRAIN, "n_test": N_TEST,
                               "r_primal": R_PRIMAL, "r_dual": R_DUAL})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
