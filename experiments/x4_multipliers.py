"""X4 multiplier variants S0, S1, S1-restricted, S2: sharpness of the certificate and cost.

For the snapped corrections R0s and Bs (gap rule, --tau-gap) at the largest primal basis size, the certificate is
evaluated with
  S0      max(lam~, 0) (lifted ROM multiplier, no optimisation),
  S1      minimum-residual multiplier, support = all n_c rows,
  S1r_A   S1-restricted on the snap set A^,
  S1r_t   S1-restricted on S_t = {g^ <= t} for t = factor * tau_gap * char_gap, factor in SUPPORT_FACTORS,
  S2      residual-only NNLS.
Records Delta, the multiplier time (the LCP / clipping only; the shared solve X^-1 a is timed separately), the support
size and whether the support covers A_+ (paper Theorem 3.6).

Criterion (fixed, plan Sec. 6): S1-restricted within a factor 1.5 of S1 at >= 5x lower cost, or reported otherwise.
Times are indicative: the LCP solver is pure Python (rom_engine.solve_lcp).

Output: results/x4/<problem>_seed<seed>.npz; arrays (correction, variant, query).
Usage:  python experiments/x4_multipliers.py --problem membrane --seed 0 [--tau-gap 1e-3]
"""

import logging
import time

import numpy as np

from certcorrect import cli
from certcorrect.active_set import by_gap, support_by_gap
from certcorrect.certificate import certify
from certcorrect.config import SEEDS, SETTINGS, TAU, TAU_LAMBDA, make_problem, result_path
from certcorrect.fom import energy_norm
from certcorrect.multipliers import clip_s0, nnls_s2
from certcorrect.pipeline import (EnergyInnerProduct, QueryEvaluator, build_bases, rom_for_size, solve_many,
                                  training_snapshots)
from certcorrect.results_io import save_result

LOG = logging.getLogger("x4_multipliers")
EXPERIMENT = "x4"
CORRECTIONS = ("R0s", "Bs")
SUPPORT_FACTORS = (1, 10, 100)
VARIANTS = ("S0", "S1", "S1r_A") + tuple(f"S1r_t{f}" for f in SUPPORT_FACTORS) + ("S2",)
DEFAULT_TAU_GAP = 1e-3
SHARPNESS_MAX = 1.5            # criterion of plan Sec. 6
SPEEDUP_MIN = 5.0


def multiplier(variant: str, ev, a, g_hat, alpha, lam_tilde, a_hat, t_unit) -> tuple[np.ndarray, np.ndarray | None]:
    """``(lam^, support or None)`` for one variant (support None = all rows)."""
    s1 = ev.s1
    if variant == "S0":
        return clip_s0(lam_tilde), None
    if variant == "S1":
        return s1.solve(a, g_hat, alpha)[0], None
    if variant == "S2":
        return nnls_s2(s1, a, alpha)[0], None
    support = a_hat if variant == "S1r_A" else support_by_gap(g_hat, int(variant.split("_t")[1]) * t_unit)
    return s1.solve(a, g_hat, alpha, support)[0], support


def run(problem, settings: dict, seed: int, tau_gap: float) -> dict[str, np.ndarray]:
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), settings["n_train"])
    r = max(settings["r_primal"])
    V, W = build_bases(snap, r, settings["r_dual"], settings["dual_kind"])
    r = min(r, V.shape[1])
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), settings["n_test"])
    refs, ok = solve_many(problem, mus, "test")
    mus = mus[ok]
    inner = EnergyInnerProduct(problem)
    ev = QueryEvaluator(problem, inner, TAU, TAU_LAMBDA)
    ev.attach_basis(V[:, :r])
    rom = rom_for_size(problem, V, W, r)
    t_unit = tau_gap * problem.char_gap
    shape = (len(CORRECTIONS), len(VARIANTS), len(mus))
    out = {key: np.full(shape, np.nan) for key in ("delta", "seconds", "support", "covers")}
    out.update({"error": np.full((len(CORRECTIONS), len(mus)), np.nan), "seconds_xinv": np.full(len(mus), np.nan),
                "norm_u": np.full(len(mus), np.nan), "r": np.array(r), "n_contact": np.array(problem.n_contact),
                "variants": np.array(VARIANTS)})
    for j, (mu, ref) in enumerate(zip(mus, refs)):
        K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
        out["norm_u"][j] = energy_norm(K, ref.u)
        red = rom.solve(mu)
        if not red.converged:
            continue
        a_hat = by_gap(problem.B @ red.u - c, tau_gap, problem.char_gap)
        alpha = inner.alpha(mu)
        for ci, name in enumerate(CORRECTIONS):
            u_hat = ev.correct(name, mu, red, c, a_hat)
            g_hat = np.maximum(problem.B @ u_hat - c, 0.0)
            a = f - K @ u_hat
            start = time.perf_counter()
            inner.solve(a)                                   # the shared solve X^-1 a, timed once per query
            out["seconds_xinv"][j] = time.perf_counter() - start
            out["error"][ci, j] = energy_norm(K, ref.u - u_hat)
            for vi, variant in enumerate(VARIANTS):
                start = time.perf_counter()
                lam, support = multiplier(variant, ev, a, g_hat, alpha, red.lam, a_hat, t_unit)
                out["seconds"][ci, vi, j] = time.perf_counter() - start
                out["delta"][ci, vi, j] = certify(K, f, problem.B, c, u_hat, lam, inner.solve, alpha).delta
                rows = np.arange(problem.n_contact) if support is None else support
                out["support"][ci, vi, j] = rows.size
                out["covers"][ci, vi, j] = float(np.isin(ref.strictly_active, rows).all())
    return out


def summarise(out: dict) -> None:
    s1 = VARIANTS.index("S1")
    for ci, name in enumerate(CORRECTIONS):
        LOG.info("%s (median effectivity of S1 %.3f):", name, np.nanmedian(out["delta"][ci, s1] / out["error"][ci]))
        for vi, variant in enumerate(VARIANTS):
            ratio = out["delta"][ci, vi] / out["delta"][ci, s1]
            speed = out["seconds"][ci, s1] / out["seconds"][ci, vi]
            meets = (np.nanmedian(ratio) <= SHARPNESS_MAX) and (np.nanmedian(speed) >= SPEEDUP_MIN)
            LOG.info("  %-8s Delta/Delta_S1 median %.3f (p90 %.3f) | speed-up vs S1 median %.1fx | support %4d | "
                     "covers A_+ %3.0f %% %s", variant, np.nanmedian(ratio), np.nanpercentile(ratio, 90),
                     np.nanmedian(speed), np.nanmedian(out["support"][ci, vi]),
                     100 * np.nanmean(out["covers"][ci, vi]),
                     "| meets X4 criterion" if variant.startswith("S1r") and meets else "")


def main(args) -> int:
    problem = make_problem(args.problem)
    settings = SETTINGS[args.problem]
    path = result_path(EXPERIMENT, args.problem, args.seed)
    LOG.info("%s: n_contact=%d, tau_gap=%g -> %s", problem.name, problem.n_contact, args.tau_gap, path)
    if args.dry_run:
        return cli.EXIT_OK
    out = run(problem, settings, args.seed, args.tau_gap)
    summarise(out)
    save_result(path, out, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed,
                            "tau_gap": args.tau_gap, "variants": list(VARIANTS), "settings": settings})
    return cli.EXIT_OK


def parser():
    p = cli.base_parser(__doc__.splitlines()[0])
    p.add_argument("--tau-gap", type=float, default=DEFAULT_TAU_GAP, help="gap-rule threshold (from X3)")
    return p


if __name__ == "__main__":
    cli.run(main, parser())
