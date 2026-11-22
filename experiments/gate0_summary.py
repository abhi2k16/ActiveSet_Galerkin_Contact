"""Gate 0 summary (FINAL_RESEARCH_PLAN.md, Sec. 7): aggregate X1 and X2 over seeds and evaluate the go criterion.

Go criterion of the plan: R0s gives a clearly better certificate rate or effectivity than R0 on at least one benchmark.
"Clearly better" is made numerical here, fixed on 22 Nov 2026 BEFORE the seed-1/2 results were seen:
  * effectivity ratio  median(Delta/e | R0) / median(Delta/e | R0s) at the largest basis size >= EFF_RATIO_MIN, or
  * rate gain          rate(R0s) - rate(R0) >= RATE_GAIN_MIN,
in EVERY seed, for at least one active-set rule. X2 criteria (plan Sec. 6) are checked per seed and mesh level.

Outputs: results/gate0/summary.json, paper/tables/gate0_x1.tex, paper/tables/gate0_x2.tex; a summary in the log.
Usage:  python experiments/gate0_summary.py [--seeds 0 1 2]
"""

import argparse
import json
import logging
import sys

import numpy as np

from certcorrect.analysis import certificate_rate
from certcorrect.cli import EXIT_BAD_INPUT, EXIT_OK, setup_logging
from certcorrect.config import EFFECTIVITY_MAX_XK, RESULTS, TABLES, result_path
from certcorrect.results_io import load_result

LOG = logging.getLogger("gate0_summary")
EFF_RATIO_MIN = 1.2
RATE_GAIN_MIN = 0.1
X1_PROBLEMS = ("membrane", "block", "layered")
X2_PROBLEMS = ("membrane", "block")
RULES = {"multiplier": "x1", "gap": "x1_gap_tg0.001"}
X2_TAG = "x2_gap_tg0.001"


def median_eff(delta: np.ndarray, error: np.ndarray, mask: np.ndarray | None = None) -> float:
    eff = delta / error
    if mask is not None:
        eff = eff[mask]
    eff = eff[np.isfinite(eff)]
    return float(np.median(eff)) if eff.size else float("nan")


def x1_metrics(a: dict) -> dict:
    """Per-seed X1 metrics; 'last' = largest basis size."""
    norm, last = a["norm_u"], -1
    m = {}
    for name in ("R0", "R0s"):
        e, d, cov = a[f"error_{name}"], a[f"delta_{name}"], a[f"covers_plus_{name}"]
        m[f"rate_{name}"] = certificate_rate(e, d)
        m[f"rate_{name}_not_covered"] = certificate_rate(e, d, cov == 0)
        m[f"eff_{name}"] = median_eff(d[last], e[last])
        m[f"err_{name}"] = float(np.nanmedian(e[last] / norm))
        m[f"acc_{name}"] = float(np.nanmean(a[f"accepted_{name}"][last]))
    m["acc_best"] = float(np.mean(a["accepted_best"][last]))
    m["delta_best"] = float(np.nanmedian(a["delta_best"][last] / norm))
    m["err_G0"] = float(np.nanmedian(a["error_G0"][last] / norm))
    m["eff_ratio"] = m["eff_R0"] / m["eff_R0s"]
    m["rate_gain"] = m["rate_R0s"] - m["rate_R0"]
    m["clearly_better"] = bool(m["eff_ratio"] >= EFF_RATIO_MIN or m["rate_gain"] >= RATE_GAIN_MIN)
    return m


def x2_metrics(a: dict) -> dict:
    """Per-seed X2 metrics per mesh level: X = K effectivity where E1(b) holds, X = I effectivity (best-of)."""
    levels = range(len(a["level_n_dof"]))
    eff_k_e1 = [median_eff(a["delta_R0s_K"][i], a["error_R0s_K"][i], a["covers_plus_R0s_K"][i] == 1) for i in levels]
    eff_i = [median_eff(a["delta_best_I"][i], a["error_best_I"][i]) for i in levels]
    eff_k = [median_eff(a["delta_best_K"][i], a["error_best_K"][i]) for i in levels]
    return {"n_dof": a["level_n_dof"].tolist(), "sqrt_condition": a["level_sqrt_condition"].tolist(),
            "eff_K_E1b": eff_k_e1, "eff_K_best": eff_k, "eff_I_best": eff_i,
            "xk_criterion": bool(np.nanmax(eff_k_e1) <= EFFECTIVITY_MAX_XK),
            "xi_grows": bool(np.all(np.diff(eff_i) > 0))}


def spread(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}


def collect(seeds: list[int]) -> dict:
    out = {"x1": {}, "x2": {}, "criteria": {"eff_ratio_min": EFF_RATIO_MIN, "rate_gain_min": RATE_GAIN_MIN,
                                           "effectivity_max_xk": EFFECTIVITY_MAX_XK}, "seeds": seeds}
    for problem in X1_PROBLEMS:
        for rule, tag in RULES.items():
            per_seed = [x1_metrics(load_result(result_path(tag, problem, s))[0]) for s in seeds]
            agg = {k: spread([m[k] for m in per_seed]) for k in per_seed[0] if k != "clearly_better"}
            agg["clearly_better_all_seeds"] = all(m["clearly_better"] for m in per_seed)
            out["x1"][f"{problem}/{rule}"] = agg
    for problem in X2_PROBLEMS:
        out["x2"][problem] = [x2_metrics(load_result(result_path(X2_TAG, problem, s))[0]) for s in seeds]
    benchmarks_go = sorted({key.split("/")[0] for key, agg in out["x1"].items() if agg["clearly_better_all_seeds"]})
    out["gate0"] = {"benchmarks_clearly_better": benchmarks_go, "go": bool(benchmarks_go),
                    "x2_xk_all": all(m["xk_criterion"] for runs in out["x2"].values() for m in runs),
                    "x2_xi_grows_all": all(m["xi_grows"] for runs in out["x2"].values() for m in runs)}
    return out


def fmt(s: dict, pct: bool = False) -> str:
    scale, unit = (100.0, r"\,\%") if pct else (1.0, "")
    if abs(s["max"] - s["min"]) * scale < (0.05 if not pct else 0.5):
        return f"{s['mean'] * scale:.2f}{unit}" if not pct else f"{s['mean'] * scale:.0f}{unit}"
    if pct:
        return f"{s['mean'] * scale:.0f}{unit} [{s['min'] * scale:.0f}, {s['max'] * scale:.0f}]"
    return f"{s['mean']:.2f} [{s['min']:.2f}, {s['max']:.2f}]"


def write_x1_table(summary: dict) -> None:
    rows = [r"\begin{tabular}{llllllll}", r"\hline",
            r"benchmark & $\hat{\mathcal A}$ rule & rate R0 & rate R0s & eff.\ R0 & eff.\ R0s & accepted R0 & "
            r"accepted best \\", r"\hline"]
    for key, agg in summary["x1"].items():
        problem, rule = key.split("/")
        rows.append(f"{problem} & {rule} & {fmt(agg['rate_R0'])} & {fmt(agg['rate_R0s'])} & {fmt(agg['eff_R0'])} & "
                    f"{fmt(agg['eff_R0s'])} & {fmt(agg['acc_R0'], True)} & {fmt(agg['acc_best'], True)} \\\\")
    rows += [r"\hline", r"\end{tabular}"]
    write_table("gate0_x1.tex", rows, summary)


def write_x2_table(summary: dict) -> None:
    rows = [r"\begin{tabular}{lrrrr}", r"\hline",
            r"benchmark & $n_{\mathrm{dof}}$ & $\sqrt{\kappa(K)}$ & eff.\ $X=K$ (E1(b)) & eff.\ $X=I$ \\", r"\hline"]
    for problem, runs in summary["x2"].items():
        for i, n_dof in enumerate(runs[0]["n_dof"]):
            ek = spread([m["eff_K_E1b"][i] for m in runs])
            ei = spread([m["eff_I_best"][i] for m in runs])
            rows.append(f"{problem} & {n_dof} & {runs[0]['sqrt_condition'][i]:.1f} & {fmt(ek)} & {fmt(ei)} \\\\")
    rows += [r"\hline", r"\end{tabular}"]
    write_table("gate0_x2.tex", rows, summary)


def write_table(name: str, rows: list[str], summary: dict) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    head = f"% generated by experiments/gate0_summary.py (seeds {summary['seeds']}); mean [min, max] over seeds\n"
    (TABLES / name).write_text(head + "\n".join(rows) + "\n", encoding="utf-8")


def log_summary(summary: dict) -> None:
    for key, agg in summary["x1"].items():
        LOG.info("X1 %-20s rate R0 %s | R0s %s | eff R0 %s | R0s %s | err R0 %s R0s %s | acc R0 %s best %s | "
                 "clearly better in all seeds: %s", key, fmt(agg["rate_R0"]), fmt(agg["rate_R0s"]),
                 fmt(agg["eff_R0"]), fmt(agg["eff_R0s"]), fmt(agg["err_R0"], True), fmt(agg["err_R0s"], True),
                 fmt(agg["acc_R0"], True), fmt(agg["acc_best"], True), agg["clearly_better_all_seeds"])
    for problem, runs in summary["x2"].items():
        for s, m in zip(summary["seeds"], runs):
            LOG.info("X2 %-8s seed %d: eff X=K (E1b) %s | X=I %s | X=K<=%.1f: %s | X=I grows: %s", problem, s,
                     np.round(m["eff_K_E1b"], 2), np.round(m["eff_I_best"], 1), EFFECTIVITY_MAX_XK,
                     m["xk_criterion"], m["xi_grows"])
    LOG.info("GATE 0: %s", summary["gate0"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    setup_logging()
    try:
        summary = collect(args.seeds)
    except FileNotFoundError as exc:
        LOG.error("missing result: %s (run experiments/run_gate0.sh first)", exc)
        return EXIT_BAD_INPUT
    out = RESULTS / "gate0" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_x1_table(summary)
    write_x2_table(summary)
    log_summary(summary)
    LOG.info("wrote %s and %s", out, TABLES)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
