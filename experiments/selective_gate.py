"""Go/no-go test of selective compliance enrichment (FINAL_RESEARCH_PLAN.md D13), parameter-dependent K only.

Layered block levels n_c = 225, 361, 529 (scaling family), r = 15, r_dual = 12, 60 training snapshots, 10 held-out
parameters (greedy selector only; no full-order solves), 30 test queries (held out from both).
Configurations, all with every n_c constraint enforced and the online certificate (S1, X = K(mu_bar), tau = 0.05):
  G                 full enrichment [V, X^-1 B^T] (baseline);
  GS-random-p       random k = p n_c rows;           GS-force-p  most frequently active training rows;
  GS-greedy-p       certificate-driven offline selection on held-out parameters;   p in BUDGETS;
  GS-pred-f         per query, ROM gap rule t = f * tau_g * char_gap (addition to the agreed protocol).
Fallback for rejected queries: warm-started full-order solve (fairest full-order reference for parameter-dependent K).
Net time per query T = T_online + (1 - share) * T_fom_warm; speed-up over G = T_G / T.

GATE (fixed 23 Nov 2026 before running, plan D13): at the largest level (n_c = 529) some configuration with
|S| <= 20 % of n_c reaches certified share >= 95 % and net speed-up over full G >= 2; and its speed-up over G does not
decrease from n_c = 361 to n_c = 529. Accepted failures (accepted with true relative error > tau) must be zero.

Output: results/selective_gate/layered_seed<seed>.npz.   Usage:  python experiments/selective_gate.py --seed 0
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before NumPy (single thread)
    os.environ[_name] = "1"

import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from certcorrect import cli  # noqa: E402
from certcorrect.config import SEEDS, TAU, TAU_GAP, result_path  # noqa: E402
from certcorrect.corrections import EnergyProjection, RightInverse  # noqa: E402
from certcorrect.fom import FullOrderSolver, energy_norm  # noqa: E402
from certcorrect.multipliers import MinResidualS1  # noqa: E402
from certcorrect.offline_online import OnlineCertifier  # noqa: E402
from certcorrect.pipeline import EnergyInnerProduct, build_bases, training_snapshots  # noqa: E402
from certcorrect.problems import ElasticBlock  # noqa: E402
from certcorrect.results_io import save_result  # noqa: E402
from certcorrect.selective import RawEnrichment, select_greedy, select_random, select_training_force  # noqa: E402

LOG = logging.getLogger("selective_gate")
EXPERIMENT = "selective_gate"
LEVELS = ((16, 4), (20, 5), (24, 6))
N_TRAIN, N_HOLD, N_TEST, R_PRIMAL, R_DUAL = 60, 10, 30, 15, 12
BUDGETS = (0.05, 0.10, 0.20)
PRED_FACTORS = (1, 10)
REPEATS = 3
HOLD_SEED_OFFSET = 50_000
GATE_SIZE, GATE_SHARE, GATE_SPEEDUP = 0.20, 0.95, 2.0


def best_time(func):
    best, out = np.inf, None
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return best, out


def run_level(n_xy: int, nz: int, seed: int) -> dict:
    problem = ElasticBlock(n_xy=n_xy, nz=nz, layered=True)
    n_c = problem.n_contact
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), N_TRAIN)
    V, W = build_bases(snap, R_PRIMAL, R_DUAL, "nmf")
    inner = EnergyInnerProduct(problem)
    Bp = RightInverse.from_constraints(problem.B)
    xinv_bt = MinResidualS1(problem.B, inner.solve).xinv_bt
    t0 = time.perf_counter()
    online_s = OnlineCertifier(problem, V, W, inner, Bp, projection=RawEnrichment(problem, V, xinv_bt, Bp))
    t_offline_s = time.perf_counter() - t0
    proj = EnergyProjection(problem, V, inner.solve, inner.matrix, Bp, "compliance", xinv_bt=xinv_bt)
    online_g = OnlineCertifier(problem, V, W, inner, Bp, projection=proj)
    hold = problem.sample(np.random.default_rng(SEEDS["train"] + seed + HOLD_SEED_OFFSET), N_HOLD)
    configs = {"G": ("G", None)}
    t_greedy = {}
    for p in BUDGETS:
        k = max(1, int(round(p * n_c)))
        configs[f"GS-random-{p:g}"] = ("GS", select_random(n_c, k, seed))
        configs[f"GS-force-{p:g}"] = ("GS", select_training_force(snap.L, k))
        t0 = time.perf_counter()
        configs[f"GS-greedy-{p:g}"] = ("GS", select_greedy(online_s, hold, snap.L, k, TAU, TAU_GAP))
        t_greedy[p] = time.perf_counter() - t0
    for f in PRED_FACTORS:
        configs[f"GS-pred-{f}"] = ("GS", "predicted", f)
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), N_TEST)
    fom = FullOrderSolver(problem)
    width = np.where(problem.hi > problem.lo, problem.hi - problem.lo, 1.0)
    out = {"n_contact": n_c, "n_dof": problem.n_dof, "t_fom_cold": np.full(N_TEST, np.nan),
           "t_fom_warm": np.full(N_TEST, np.nan), "norm_u": np.full(N_TEST, np.nan), "t_offline_s": t_offline_s,
           "t_greedy": t_greedy, "configs": list(configs)}
    for name in configs:
        for key in ("t", "error", "size"):
            out[f"{key}:{name}"] = np.full(N_TEST, np.nan)
        out[f"accepted:{name}"] = np.zeros(N_TEST, bool)
        out[f"feasible:{name}"] = np.ones(N_TEST, bool)
    for j, mu in enumerate(mus):
        K = problem.stiffness(mu)
        start = time.perf_counter()
        ref = fom.solve(mu, K=K)
        out["t_fom_cold"][j] = time.perf_counter() - start
        if not ref.converged:
            continue
        nearest = int(np.argmin(np.linalg.norm((snap.mus - mu) / width, axis=1)))
        mask = np.zeros(n_c, bool)
        mask[snap.active[nearest]] = True
        start = time.perf_counter()
        fom.solve(mu, active0=mask, K=K)
        out["t_fom_warm"][j] = time.perf_counter() - start
        out["norm_u"][j] = energy_norm(K, ref.u)
        for name, spec in configs.items():
            if spec[0] == "G":
                cert, fn = online_g, (lambda: online_g.evaluate(mu, TAU, TAU_GAP, correction="G"))
            else:
                factor = spec[2] if len(spec) > 2 else 1.0
                cert = online_s
                fn = (lambda s=spec[1], fac=factor: online_s.evaluate(mu, TAU, TAU_GAP, correction="GS", select=s,
                                                                       predicted_factor=fac))
            t, res = best_time(fn)
            out[f"t:{name}"][j], out[f"accepted:{name}"][j] = t, res.accepted
            out[f"error:{name}"][j] = energy_norm(K, ref.u - cert.displacement(res))
            out[f"size:{name}"][j] = n_c if spec[0] == "G" else len(res.selected)
            out[f"feasible:{name}"][j] = res.feasible
    return out


def summarise_level(lv: dict) -> dict:
    ok = np.isfinite(lv["norm_u"])
    t_fb = float(np.nanmedian(lv["t_fom_warm"]))
    rows = {}
    for name in lv["configs"]:
        share = float(np.mean(lv[f"accepted:{name}"][ok]))
        t_on = float(np.nanmedian(lv[f"t:{name}"]))
        rel = lv[f"error:{name}"][ok] / lv["norm_u"][ok]
        rows[name] = {"share": share, "t_online": t_on, "t_net": t_on + (1 - share) * t_fb,
                      "error": float(np.median(rel)), "size_frac": float(np.nanmean(lv[f"size:{name}"]) / lv["n_contact"]),
                      "feasible": float(np.mean(lv[f"feasible:{name}"][ok])),
                      "accepted_failures": int(np.sum(lv[f"accepted:{name}"][ok] & (rel > TAU)))}
    for name in rows:
        rows[name]["speedup_vs_G"] = rows["G"]["t_net"] / rows[name]["t_net"]
        rows[name]["speedup_vs_fom_warm"] = t_fb / rows[name]["t_net"]
    return {"t_fom_warm": t_fb, "t_fom_cold": float(np.nanmedian(lv["t_fom_cold"])), "rows": rows}


def gate(summaries: list[dict]) -> tuple[bool, list[str]]:
    last, prev = summaries[-1]["rows"], summaries[-2]["rows"]
    passing = [n for n, r in last.items() if n != "G" and r["size_frac"] <= GATE_SIZE and r["share"] >= GATE_SHARE
               and r["speedup_vs_G"] >= GATE_SPEEDUP and r["speedup_vs_G"] >= prev[n]["speedup_vs_G"]
               and r["accepted_failures"] == 0]
    return bool(passing), passing


def main(args) -> int:
    path = result_path(EXPERIMENT, "layered", args.seed)
    LOG.info("levels %s, budgets %s, predicted factors %s -> %s", LEVELS, BUDGETS, PRED_FACTORS, path)
    if args.dry_run:
        return cli.EXIT_OK
    arrays, summaries = {}, []
    for k, (n_xy, nz) in enumerate(LEVELS):
        lv = run_level(n_xy, nz, args.seed)
        sm = summarise_level(lv)
        summaries.append({"n_contact": int(lv["n_contact"]), "n_dof": int(lv["n_dof"]), **sm,
                          "t_greedy": {str(p): v for p, v in lv["t_greedy"].items()}})
        LOG.info("n_c %d (N %d): FOM cold %.0f ms, warm %.0f ms", lv["n_contact"], lv["n_dof"], 1e3 * sm["t_fom_cold"],
                 1e3 * sm["t_fom_warm"])
        for name, r in sm["rows"].items():
            LOG.info("  %-18s |S|/n_c %5.1f %% | cert %5.1f %% (fails %d, QP feasible %3.0f %%) | err %.3f %% | online "
                     "%6.2f ms | net %6.2f ms | vs G %5.2fx | vs FOM-warm %6.1fx", name, 100 * r["size_frac"],
                     100 * r["share"], r["accepted_failures"], 100 * r["feasible"], 100 * r["error"],
                     1e3 * r["t_online"], 1e3 * r["t_net"], r["speedup_vs_G"], r["speedup_vs_fom_warm"])
        for key, val in lv.items():
            if isinstance(val, np.ndarray):
                arrays[f"L{k}|{key}"] = val
    passed, passing = gate(summaries)
    LOG.info("GATE (n_c = %d): %s%s", summaries[-1]["n_contact"], "PASS" if passed else "FAIL",
             f" — passing configurations: {passing}" if passed else "")
    save_result(path, arrays, {"experiment": EXPERIMENT, "seed": args.seed, "levels": LEVELS, "summaries": summaries,
                               "gate": {"passed": passed, "passing": passing, "size": GATE_SIZE, "share": GATE_SHARE,
                                        "speedup": GATE_SPEEDUP}})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
