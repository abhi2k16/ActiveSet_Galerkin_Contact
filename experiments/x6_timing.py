"""X6 full pipeline timing and break-even against FOM-cold and FOM-cond (Gate 1).

Online pipeline (certcorrect.offline_online): ROM solve -> R0s (gap rule, config.TAU_GAP) -> multiplier (S1, and
S1-restricted on S_t with t = config.SUPPORT_FACTOR * TAU_GAP * char_gap) -> certificate and acceptance at tau. A rejected
query falls back to a full-order solve, so the expected time per query is
    T_pipe = T_online + (1 - share) * T_fom,    net speed-up = T_fom / T_pipe,
and the break-even number of queries is (offline cost) / (T_fom - T_pipe), with the offline cost = training full-order
solves + bases + representers (OnlineCertifier). FOM-cold: assembly of K(mu) and the active-set solver from scratch;
FOM-cond: reused factorisation and Delassus matrix (constant K only).

Gate 1 (plan Sec. 7, fixed): speed claim if net speed-up > 1 with certified share >= 50 % on the layered block.
BLAS is pinned to one thread (environment.single_thread) before NumPy is imported.

Output: results/x6/<problem>_seed<seed>.npz.   Usage:  python experiments/x6_timing.py --problem layered --seed 0
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before NumPy (single thread)
    os.environ[_name] = "1"

import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from certcorrect import cli  # noqa: E402
from certcorrect.config import (GATE1_MIN_CERTIFIED_SHARE, SEEDS, SETTINGS, SUPPORT_FACTOR, TAU, TAU_GAP,  # noqa: E402
                                make_problem, result_path)
from certcorrect.fom import CondensedSolver, FullOrderSolver, energy_norm  # noqa: E402
from certcorrect.offline_online import STAGES, OnlineCertifier  # noqa: E402
from certcorrect.corrections import EnergyProjection, RightInverse  # noqa: E402
from certcorrect.multipliers import MinResidualS1  # noqa: E402
from certcorrect.pipeline import EnergyInnerProduct, build_bases, training_snapshots  # noqa: E402
from certcorrect.results_io import save_result  # noqa: E402

LOG = logging.getLogger("x6_timing")
EXPERIMENT = "x6"
VARIANTS = {"S1": None, "S1r": SUPPORT_FACTOR}
REPEATS = 3                                   # best-of-3 wall time per query and variant


def best_time(func, repeats=REPEATS):
    best, out = np.inf, None
    for _ in range(repeats):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return best, out


def run(problem, settings: dict, seed: int, correction: str = "R0s") -> dict:
    t0 = time.perf_counter()
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), settings["n_train"])
    t_train = time.perf_counter() - t0
    r = max(settings["r_primal"])
    t0 = time.perf_counter()
    V, W = build_bases(snap, r, settings["r_dual"], settings["dual_kind"])
    t_bases = time.perf_counter() - t0
    inner = EnergyInnerProduct(problem)
    projection = None
    if correction == "Bs":                         # projection basis Z and its offline terms count as offline cost
        t0 = time.perf_counter()
        projection = EnergyProjection(problem, V, inner.solve, inner.matrix, RightInverse.from_constraints(problem.B),
                                      "compliance", xinv_bt=MinResidualS1(problem.B, inner.solve).xinv_bt)
        t_bases += time.perf_counter() - t0
    online = OnlineCertifier(problem, V, W, inner, projection=projection)
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), settings["n_test"])
    fom = FullOrderSolver(problem)
    cond = None if problem.k_varies else CondensedSolver(problem)
    n = len(mus)
    out = {"t_train": np.array(t_train), "t_bases": np.array(t_bases), "t_representers": np.array(online.offline_seconds),
           "t_cond_offline": np.array(np.nan if cond is None else cond.offline_seconds),
           "t_fom_cold": np.full(n, np.nan), "t_fom_cond": np.full(n, np.nan), "error": np.full(n, np.nan),
           "norm_u": np.full(n, np.nan), "n_dof": np.array(problem.n_dof), "n_contact": np.array(problem.n_contact),
           "r": np.array(V.shape[1]), "n_columns": np.array(online.n_columns),
           "affine_load": np.array(online.is_affine_load)}
    for v in VARIANTS:
        out[f"t_online_{v}"] = np.full(n, np.nan)
        out[f"accepted_{v}"] = np.zeros(n, bool)
        out[f"delta_{v}"] = np.full(n, np.nan)
        for s in STAGES:
            out[f"t_{s}_{v}"] = np.full(n, np.nan)
    for j, mu in enumerate(mus):
        t_cold, ref = best_time(lambda: fom.solve(mu, K=problem.stiffness(mu)), repeats=1)
        out["t_fom_cold"][j] = t_cold
        if not ref.converged:
            continue
        K = problem.stiffness(mu)
        out["norm_u"][j] = energy_norm(K, ref.u)
        if cond is not None:
            out["t_fom_cond"][j] = best_time(lambda: cond.solve(mu))[0]
        for v, factor in VARIANTS.items():
            t_on, res = best_time(lambda: online.evaluate(mu, TAU, TAU_GAP, support_factor=factor,
                                                          correction=correction))
            out[f"t_online_{v}"][j] = t_on
            out[f"accepted_{v}"][j] = res.accepted
            out[f"delta_{v}"][j] = res.delta
            for s in STAGES:
                out[f"t_{s}_{v}"][j] = res.seconds[s]
            if v == "S1":
                out["error"][j] = energy_norm(K, ref.u - online.displacement(res))
    return out


def summarise(out: dict, problem_name: str) -> dict:
    offline = float(out["t_train"] + out["t_bases"] + out["t_representers"])
    summary = {"offline_seconds": offline}
    t_cold = float(np.nanmedian(out["t_fom_cold"]))
    t_cond = float(np.nanmedian(out["t_fom_cond"])) if np.isfinite(out["t_fom_cond"]).any() else float("nan")
    ok = np.isfinite(out["error"])
    silent = np.sum(out["accepted_S1"] & ok & (out["error"] > TAU * out["norm_u"]))
    LOG.info("%s: n_dof %d, n_c %d, r %d, columns %d, load %s | offline: training %.1f s, bases %.1f s, representers "
             "%.1f s | FOM-cold %.1f ms, FOM-cond %s ms", problem_name, out["n_dof"], out["n_contact"], out["r"],
             out["n_columns"], "affine" if out["affine_load"] else "hybrid (one solve per query)", out["t_train"],
             out["t_bases"], out["t_representers"], 1e3 * t_cold, f"{1e3 * t_cond:.2f}" if np.isfinite(t_cond) else "n/a")
    LOG.info("accepted silent failures (accepted with true relative error > tau): %d", silent)
    for v in VARIANTS:
        share = float(np.mean(out[f"accepted_{v}"][ok]))
        t_on = float(np.nanmedian(out[f"t_online_{v}"]))
        stages = ", ".join(f"{s} {1e3 * np.nanmedian(out[f't_{s}_{v}']):.2f}" for s in STAGES)
        for ref_name, t_fom in (("cold", t_cold), ("cond", t_cond)):
            if not np.isfinite(t_fom):
                continue
            t_pipe = t_on + (1 - share) * t_fom
            speedup = t_fom / t_pipe
            gain = t_fom - t_pipe
            breakeven = offline / gain if gain > 0 else float("inf")
            summary[f"{v}_{ref_name}"] = {"share": share, "t_online": t_on, "t_fom": t_fom, "speedup": speedup,
                                          "breakeven_queries": breakeven}
            LOG.info("%-3s vs FOM-%s: certified share %.0f %% | online %.2f ms (%s) | net speed-up %.2fx | break-even "
                     "after %.0f queries", v, ref_name, 100 * share, 1e3 * t_on, stages, speedup, breakeven)
    if problem_name == "layered":
        s = summary["S1_cold"]
        LOG.info("GATE 1 (layered): net speed-up %.2f > 1 and certified share %.0f %% >= %.0f %%: %s", s["speedup"],
                 100 * s["share"], 100 * GATE1_MIN_CERTIFIED_SHARE,
                 s["speedup"] > 1 and s["share"] >= GATE1_MIN_CERTIFIED_SHARE)
    return summary


def main(args) -> int:
    problem = make_problem(args.problem)
    settings = SETTINGS[args.problem]
    path = result_path(EXPERIMENT if args.correction == "R0s" else f"{EXPERIMENT}_{args.correction}", args.problem,
                       args.seed)
    LOG.info("%s -> %s (threads pinned to 1)", problem.name, path)
    if args.dry_run:
        return cli.EXIT_OK
    out = run(problem, settings, args.seed, args.correction)
    summary = summarise(out, args.problem)
    save_result(path, out, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed, "tau": TAU,
                            "tau_gap": TAU_GAP, "support_factor": SUPPORT_FACTOR, "summary": summary,
                            "settings": settings, "correction": args.correction})
    return cli.EXIT_OK


def parser():
    p = cli.base_parser(__doc__.splitlines()[0])
    p.add_argument("--correction", choices=("R0s", "Bs"), default="R0s")
    return p


if __name__ == "__main__":
    cli.run(main, parser())
