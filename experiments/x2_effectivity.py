"""X2 effectivity versus error and versus mesh size h, X = I versus X = K.

Criterion (fixed, plan Sec. 6): X = K effectivity <= 1.2 when Corollary E1(b) holds (here: ``A_+`` contained in the
contact set of ``u^``); X = I effectivity grows with mesh refinement.

Per mesh level (config.X2_LEVELS) the same pipeline as X1 runs at one basis size: snapshots, G0, corrections R0 and
R0s, the S1 multiplier for each inner product, and the certificate with
  * ``X = K``, ``alpha_X = 1``;
  * ``X = I``, ``alpha_X`` = Lanczos estimate of ``lambda_min(K)`` (UNVERIFIED; favours the baseline, see verified.py).
Output: ``results/x2[...]/<problem>_seed<seed>.npz``; per-query arrays have shape (number of levels, number of tests).

Usage:  python experiments/x2_effectivity.py --problem membrane --seed 0 [--rule gap --tau-gap 1e-3]
"""

import logging

import numpy as np

from certcorrect import cli
from certcorrect.analysis import best_of
from certcorrect.config import SEEDS, TAU, TAU_GAP, TAU_LAMBDA, X2_LEVELS, X2_SETTINGS, result_path
from certcorrect.fom import energy_norm
from certcorrect.pipeline import (ACTIVE_SET_RULES, CORRECTIONS, EnergyInnerProduct, EuclideanInnerProduct,
                                  QueryEvaluator, build_bases, rom_for_size, solve_many, training_snapshots)
from certcorrect.results_io import save_result
from certcorrect.verified import extreme_eigenvalues

LOG = logging.getLogger("x2_effectivity")
EXPERIMENT = "x2"
INNER_PRODUCTS = ("K", "I")
FIELDS = ("error", "delta", "accepted", "covers_plus")


def run_level(problem, settings: dict, seed: int, rule: str, tau_lambda: float, tau_gap: float) -> dict:
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), settings["n_train"])
    V, W = build_bases(snap, settings["r_primal"], settings["r_dual"], settings["dual_kind"])
    rom = rom_for_size(problem, V, W, V.shape[1])
    test_mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), settings["n_test"])
    refs, ok = solve_many(problem, test_mus, f"test (n_dof={problem.n_dof})")
    test_mus = test_mus[ok]
    eig = extreme_eigenvalues(problem.stiffness(problem.lo))
    evaluators = {"K": QueryEvaluator(problem, EnergyInnerProduct(problem), TAU, tau_lambda, rule, tau_gap),
                  "I": QueryEvaluator(problem, EuclideanInnerProduct(problem, eig.lambda_min), TAU, tau_lambda, rule,
                                      tau_gap)}
    n = settings["n_test"]
    out = {"norm_u": np.full(n, np.nan), "error_G0": np.full(n, np.nan), "rom_ok": np.zeros(n, bool)}
    for x in INNER_PRODUCTS:
        for name in CORRECTIONS:
            for field in FIELDS:
                out[f"{field}_{name}_{x}"] = np.full(n, np.nan)
    for j, (mu, ref) in enumerate(zip(test_mus, refs)):
        K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
        out["norm_u"][j] = energy_norm(K, ref.u)
        red = rom.solve(mu)
        out["rom_ok"][j] = red.converged
        if not red.converged:
            continue
        out["error_G0"][j] = energy_norm(K, ref.u - red.u)
        active_hat = evaluators["K"].predicted_active(red, c)
        for x, ev in evaluators.items():
            for name in CORRECTIONS:
                cert = ev.certify_correction(name, mu, K, f, c, ref, red, active_hat)
                for field in FIELDS:
                    out[f"{field}_{name}_{x}"][j] = getattr(cert, field)
    for x in INNER_PRODUCTS:
        for key, value in best_of(out, CORRECTIONS, suffix=f"_{x}").items():
            out[f"{key}_best_{x}"] = value
    scalars = {"n_dof": problem.n_dof, "n_contact": problem.n_contact, "h": 1.0 / level_size(problem),
               "lambda_min": eig.lambda_min, "lambda_max": eig.lambda_max, "sqrt_condition": eig.sqrt_condition,
               "ritz_residual": eig.residual_min, "r_primal": V.shape[1], "r_dual": W.shape[1]}
    return out, scalars


def level_size(problem) -> int:
    return problem.n if hasattr(problem, "n") else problem.n_xy


def effectivity(out: dict, method: str, x: str, mask: np.ndarray | None = None) -> float:
    eff = out[f"delta_{method}_{x}"] / out[f"error_{method}_{x}"]
    if mask is not None:
        eff = eff[mask]
    eff = eff[np.isfinite(eff)]
    return float(np.median(eff)) if eff.size else float("nan")


def stack_levels(levels: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.stack([lv[key] for lv in levels]) for key in levels[0]}


def output_path(problem: str, seed: int, rule: str, tau_lambda: float, tau_gap: float):
    tag = EXPERIMENT
    if rule != "multiplier" or tau_lambda != TAU_LAMBDA:
        tag += f"_{rule}_tl{tau_lambda:g}" if rule == "multiplier" else f"_{rule}_tg{tau_gap:g}"
    return result_path(tag, problem, seed)


def main(args) -> int:
    if args.problem not in X2_LEVELS:
        raise ValueError(f"X2 is defined for {sorted(X2_LEVELS)} (constant K), not {args.problem!r}")
    settings = X2_SETTINGS[args.problem]
    path = output_path(args.problem, args.seed, args.rule, args.tau_lambda, args.tau_gap)
    LOG.info("%s: %d mesh levels, settings %s, rule=%s -> %s", args.problem, len(X2_LEVELS[args.problem]), settings,
             args.rule, path)
    if args.dry_run:
        return cli.EXIT_OK
    per_level, scalars = [], []
    for make in X2_LEVELS[args.problem]:
        out, sc = run_level(make(), settings, args.seed, args.rule, args.tau_lambda, args.tau_gap)
        per_level.append(out)
        scalars.append(sc)
        covers = out["covers_plus_R0s_K"] == 1
        LOG.info("n_dof=%5d sqrt(kappa)=%7.1f | median eff. R0: K %.2f, I %.1f | R0s: K %.2f (E1(b) holds: %.2f, "
                 "n=%d), I %.1f | best: K %.2f, I %.1f", sc["n_dof"], sc["sqrt_condition"],
                 effectivity(out, "R0", "K"), effectivity(out, "R0", "I"), effectivity(out, "R0s", "K"),
                 effectivity(out, "R0s", "K", covers), covers.sum(), effectivity(out, "R0s", "I"),
                 effectivity(out, "best", "K"), effectivity(out, "best", "I"))
    arrays = stack_levels(per_level)
    arrays.update({f"level_{key}": np.array([sc[key] for sc in scalars]) for key in scalars[0]})
    save_result(path, arrays, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed, "tau": TAU,
                               "rule": args.rule, "tau_lambda": args.tau_lambda, "tau_gap": args.tau_gap,
                               "settings": settings, "alpha_I": "Lanczos estimate of lambda_min(K) (unverified)"})
    return cli.EXIT_OK


def parser():
    p = cli.base_parser(__doc__.splitlines()[0])
    p.add_argument("--rule", choices=ACTIVE_SET_RULES, default="multiplier", help="predicted active-set rule")
    p.add_argument("--tau-lambda", type=float, default=TAU_LAMBDA)
    p.add_argument("--tau-gap", type=float, default=TAU_GAP)
    return p


if __name__ == "__main__":
    cli.run(main, parser())
