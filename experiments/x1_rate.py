"""X1 rate test: certificate and true error against primal basis size, R0 versus R0s, multiplier S1, X = K.

Criterion (fixed, plan Sec. 6): the R0 certificate slope is about half the error slope where ``A_+`` is not contained
in the contact set of ``u^``; the R0s slope is about the error slope. Here "slope" is the rate of ``log Delta`` in
``log ||e||_K`` over all queries and basis sizes (1 = full rate, 1/2 = square-root rate).

Output: ``results/x1/<problem>_seed<seed>.npz``; arrays have shape (number of basis sizes, number of test queries).
The figure comes from ``figures/fig_x1_rate.py`` (reads the file only).

Usage:  python experiments/x1_rate.py --problem membrane --seed 0
"""

import logging

import numpy as np

from certcorrect import cli
from certcorrect.analysis import best_of, certificate_rate, median_by_row
from certcorrect.config import SEEDS, SETTINGS, TAU, TAU_GAP, TAU_LAMBDA, make_problem, result_path
from certcorrect.fom import energy_norm
from certcorrect.pipeline import (ACTIVE_SET_RULES, ALL_CORRECTIONS, CORRECTIONS, EnergyInnerProduct, QueryEvaluator,
                                  build_bases, rom_for_size, solve_many, training_snapshots)
from certcorrect.results_io import save_result

LOG = logging.getLogger("x1_rate")
EXPERIMENT = "x1"
DEFAULT_RULE = "multiplier"
FIELDS = ("error", "delta", "rho_bar", "complementarity", "accepted", "covers_plus", "s1_ok", "seconds_correction",
          "seconds_s1")


def run(problem, settings: dict, seed: int, rule: str, tau_lambda: float, tau_gap: float,
        corrections: tuple[str, ...] = CORRECTIONS) -> dict[str, np.ndarray]:
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), settings["n_train"])
    r_values = np.array([r for r in settings["r_primal"] if r <= snap.U.shape[1]])
    V, W = build_bases(snap, int(r_values.max()), settings["r_dual"], settings["dual_kind"])
    r_values = r_values[r_values <= V.shape[1]]
    test_mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), settings["n_test"])
    refs, ok = solve_many(problem, test_mus, "test")
    test_mus = test_mus[ok]
    inner = EnergyInnerProduct(problem)
    evaluator = QueryEvaluator(problem, inner, TAU, tau_lambda, rule, tau_gap)
    shape = (len(r_values), len(test_mus))
    out = {"r_values": r_values, "mus": test_mus, "r_dual": np.array(W.shape[1]),
           "norm_u": np.array([energy_norm(problem.stiffness(mu), ref.u) for mu, ref in zip(test_mus, refs)]),
           "n_strictly_active": np.array([len(ref.strictly_active) for ref in refs]),
           "error_G0": np.full(shape, np.nan), "rom_ok": np.zeros(shape, bool), "rom_seconds": np.full(shape, np.nan),
           "recall": np.full(shape, np.nan), "oversnap": np.full(shape, np.nan)}
    for name in corrections:
        for field in FIELDS:
            out[f"{field}_{name}"] = np.full(shape, np.nan)
    for i, r in enumerate(r_values):
        rom = rom_for_size(problem, V, W, int(r))
        if {"B", "Bs"} & set(corrections):
            evaluator.attach_basis(V[:, :int(r)])
        for j, (mu, ref) in enumerate(zip(test_mus, refs)):
            K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
            red = rom.solve(mu)
            out["rom_ok"][i, j], out["rom_seconds"][i, j] = red.converged, red.seconds
            if not red.converged:                       # no prediction to correct; counted, not certified
                continue
            out["error_G0"][i, j] = energy_norm(K, ref.u - red.u)
            out["recall"][i, j], out["oversnap"][i, j] = evaluator.active_set_quality(red, ref, c)
            active_hat = evaluator.predicted_active(red, c)
            for name in corrections:
                cert = evaluator.certify_correction(name, mu, K, f, c, ref, red, active_hat)
                for field in FIELDS:
                    out[f"{field}_{name}"][i, j] = getattr(cert, field)
        log_basis_summary(out, i, int(r), corrections)
    return out


def log_basis_summary(out: dict, i: int, r: int, corrections: tuple[str, ...]) -> None:
    rel = lambda e: np.nanmedian(e[i] / out["norm_u"])          # noqa: E731
    parts = [f"{m} err {rel(out[f'error_{m}']):.2e} eff {np.nanmedian(out[f'delta_{m}'][i] / out[f'error_{m}'][i]):.2f}"
             for m in corrections]
    LOG.info("r=%3d: ROM ok %d/%d | median rel. error G0 %.2e | %s", r, out["rom_ok"][i].sum(), out["rom_ok"].shape[1],
             rel(out["error_G0"]), " | ".join(parts))


def add_best(out: dict, corrections: tuple[str, ...]) -> None:
    """Per query, keep the correction with the smallest certificate (``best``); see ``analysis.best_of``."""
    for key, value in best_of(out, corrections).items():
        out[f"{key}_best"] = value


def log_rates(out: dict, corrections: tuple[str, ...]) -> None:
    norm = out["norm_u"][None, :]
    for name in corrections:
        covers = out[f"covers_plus_{name}"] == 1
        LOG.info("%-4s certificate rate: all %.2f | A_+ covered %.2f (%d) | not covered %.2f (%d) | accepted %.0f %%",
                 name, certificate_rate(out[f"error_{name}"], out[f"delta_{name}"]),
                 certificate_rate(out[f"error_{name}"], out[f"delta_{name}"], covers), covers.sum(),
                 certificate_rate(out[f"error_{name}"], out[f"delta_{name}"], out[f"covers_plus_{name}"] == 0),
                 (out[f"covers_plus_{name}"] == 0).sum(), 100 * np.nanmean(out[f"accepted_{name}"]))
    chosen = out["chosen_best"]
    shares = ", ".join(f"{m} {100 * np.mean(chosen[chosen >= 0] == k):.0f} %" for k, m in enumerate(corrections))
    LOG.info("best certificate rate %.2f | accepted %.0f %% | chosen: %s | median rel. Delta by r: %s",
             certificate_rate(out["error_best"], out["delta_best"]), 100 * np.mean(out["accepted_best"]), shares,
             np.array2string(median_by_row(out["delta_best"] / norm), precision=3))


def output_path(problem: str, seed: int, rule: str, tau_lambda: float, tau_gap: float, r_dual: int | None,
                corrections: tuple[str, ...] = CORRECTIONS):
    """Default rule, threshold, dual size and corrections -> ``results/x1/``; any other choice gets its own folder."""
    tag = EXPERIMENT
    if rule != DEFAULT_RULE or tau_lambda != TAU_LAMBDA:
        tag += f"_{rule}_tl{tau_lambda:g}" if rule == "multiplier" else f"_{rule}_tg{tau_gap:g}"
    if r_dual is not None:
        tag += f"_rd{r_dual}"
    if tuple(corrections) != CORRECTIONS:
        tag += "_" + "-".join(corrections)
    return result_path(tag, problem, seed)


def main(args) -> int:
    problem = make_problem(args.problem)
    settings = dict(SETTINGS[args.problem])
    if args.r_dual is not None:
        if args.r_dual < 1:
            raise ValueError("--r-dual must be >= 1")
        settings["r_dual"] = args.r_dual
    corrections = tuple(args.corrections)
    path = output_path(args.problem, args.seed, args.rule, args.tau_lambda, args.tau_gap, args.r_dual, corrections)
    LOG.info("%s: n_dof=%d, n_contact=%d, r_primal=%s, r_dual=%d (%s), rule=%s (tau_lambda=%g, tau_gap=%g) -> %s",
             problem.name, problem.n_dof, problem.n_contact, settings["r_primal"], settings["r_dual"],
             settings["dual_kind"], args.rule, args.tau_lambda, args.tau_gap, path)
    if args.dry_run:
        return cli.EXIT_OK
    out = run(problem, settings, args.seed, args.rule, args.tau_lambda, args.tau_gap, corrections)
    add_best(out, corrections)
    log_rates(out, corrections)
    save_result(path, out, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed, "tau": TAU,
                            "tau_lambda": args.tau_lambda, "tau_gap": args.tau_gap, "settings": settings,
                            "inner_product": "X = K(mu_bar)", "multiplier": "S1", "active_set_rule": args.rule,
                            "corrections": list(corrections), "projection_basis": "compliance"})
    return cli.EXIT_OK


def parser():
    p = cli.base_parser(__doc__.splitlines()[0])
    p.add_argument("--rule", choices=ACTIVE_SET_RULES, default=DEFAULT_RULE, help="predicted active-set rule")
    p.add_argument("--tau-lambda", type=float, default=TAU_LAMBDA, help="relative multiplier threshold")
    p.add_argument("--tau-gap", type=float, default=TAU_GAP, help="gap threshold in units of char_gap")
    p.add_argument("--r-dual", type=int, default=None, help="override the dual basis size of config.SETTINGS")
    p.add_argument("--corrections", nargs="+", choices=ALL_CORRECTIONS, default=list(CORRECTIONS),
                   help="corrections to evaluate (B / Bs: energy projection, compliance basis)")
    return p


if __name__ == "__main__":
    cli.run(main, parser())
