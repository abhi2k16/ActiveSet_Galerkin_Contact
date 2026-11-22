"""X3 active-set prediction: recall of A_+, over-snap share and certified share for the A^ rules.

Rules: multiplier (tau_lambda), gap (tau_g sweep), both (gap and multiplier), neighbours (k nearest training
parameters). For each rule the snapped corrections R0s and Bs are certified with S1, X = K(mu_bar); R0 and B (no
snap) are evaluated once per query as references. Also records the missed rows M = {i in A_+ : g^_i > 0} of R0 and B
(hypothesis of week 5: unsnapped B creates more missed rows than R0).

Criterion (plan Sec. 6; selection rule fixed 23 Nov 2026 before running, FINAL_RESEARCH_PLAN.md Sec. 12a): report the
trade-off; the rule and tau_g are chosen by the certified share of best-of(R0, R0s, Bs) at the largest basis size,
averaged over benchmarks and seeds (experiments/x3_select.py); ties go to the smaller tau_g.

Output: results/x3/<problem>_seed<seed>.npz; arrays (rule, basis size, query).
Usage:  python experiments/x3_active_set.py --problem membrane --seed 0
"""

import logging

import numpy as np

from certcorrect import cli
from certcorrect.active_set import by_gap, by_multiplier, by_neighbours, recall_and_oversnap
from certcorrect.analysis import best_of
from certcorrect.config import SEEDS, SETTINGS, TAU, TAU_LAMBDA, make_problem, result_path
from certcorrect.fom import energy_norm
from certcorrect.pipeline import (CONTACT_GAP_RTOL, EnergyInnerProduct, QueryEvaluator, build_bases, rom_for_size,
                                  solve_many, training_snapshots)
from certcorrect.results_io import save_result

LOG = logging.getLogger("x3_active_set")
EXPERIMENT = "x3"
TAU_GAPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
BOTH_TAU_GAP = 1e-2
NEIGHBOURS = (1, 3, 5)
N_SIZES = 2                                   # the two largest primal basis sizes of config.SETTINGS
SNAPPED = ("R0s", "Bs")
UNSNAPPED = ("R0", "B")
FIELDS = ("error", "delta", "accepted")


def rule_list() -> list[tuple[str, str, float]]:
    """``(label, kind, parameter)`` for every rule evaluated."""
    rules = [("multiplier", "multiplier", TAU_LAMBDA)]
    rules += [(f"gap_{t:g}", "gap", t) for t in TAU_GAPS]
    rules += [(f"both_{BOTH_TAU_GAP:g}", "both", BOTH_TAU_GAP)]
    rules += [(f"neighbours_{k}", "neighbours", k) for k in NEIGHBOURS]
    return rules


def predict(kind: str, param: float, red, g_tilde, problem, mu, snap) -> np.ndarray:
    if kind == "multiplier":
        return by_multiplier(red.lam, param)
    if kind == "gap":
        return by_gap(g_tilde, param, problem.char_gap)
    if kind == "both":
        return np.intersect1d(by_gap(g_tilde, param, problem.char_gap), by_multiplier(red.lam, TAU_LAMBDA))
    if kind == "neighbours":
        return by_neighbours(mu, snap.mus, snap.active, int(param), problem.lo, problem.hi)
    raise ValueError(f"unknown rule kind {kind!r}")


def missed(lam: np.ndarray, g_hat: np.ndarray, tol: float) -> tuple[int, float]:
    """``|M|`` and ``||lam_M||_1`` with ``M = {i : lam_i > 0, g^_i > tol}``."""
    rows = (lam > 0) & (g_hat > tol)
    return int(rows.sum()), float(lam[rows].sum())


def run(problem, settings: dict, seed: int) -> dict[str, np.ndarray]:
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), settings["n_train"])
    r_values = np.array(sorted(settings["r_primal"])[-N_SIZES:])
    V, W = build_bases(snap, int(r_values.max()), settings["r_dual"], settings["dual_kind"])
    r_values = r_values[r_values <= V.shape[1]]
    mus = problem.sample_test(np.random.default_rng(SEEDS["test"] + seed), settings["n_test"])
    refs, ok = solve_many(problem, mus, "test")
    mus = mus[ok]
    ev = QueryEvaluator(problem, EnergyInnerProduct(problem), TAU, TAU_LAMBDA)
    tol = CONTACT_GAP_RTOL * problem.char_gap
    rules = rule_list()
    n_rule, n_r, n_q = len(rules), len(r_values), len(mus)
    out = {"r_values": r_values, "rule_labels": np.array([r[0] for r in rules]),
           "norm_u": np.array([energy_norm(problem.stiffness(mu), ref.u) for mu, ref in zip(mus, refs)])}
    for name in SNAPPED:
        for field in FIELDS:
            out[f"{field}_{name}"] = np.full((n_rule, n_r, n_q), np.nan)
    for key in ("recall", "oversnap", "size"):
        out[key] = np.full((n_rule, n_r, n_q), np.nan)
    for name in UNSNAPPED:
        for field in FIELDS + ("n_missed", "lam_missed"):
            out[f"{field}_{name}"] = np.full((n_r, n_q), np.nan)
    for i, r in enumerate(r_values):
        rom = rom_for_size(problem, V, W, int(r))
        ev.attach_basis(V[:, :int(r)])
        for j, (mu, ref) in enumerate(zip(mus, refs)):
            K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
            red = rom.solve(mu)
            if not red.converged:
                continue
            g_tilde = problem.B @ red.u - c
            empty = np.zeros(0, dtype=int)
            for name in UNSNAPPED:
                cert = ev.certify_correction(name, mu, K, f, c, ref, red, empty)
                u_hat = ev.correct(name, mu, red, c, empty)
                n_m, lam_m = missed(ref.lam, problem.B @ u_hat - c, tol)
                for field, value in (("error", cert.error), ("delta", cert.delta), ("accepted", cert.accepted),
                                     ("n_missed", n_m), ("lam_missed", lam_m)):
                    out[f"{field}_{name}"][i, j] = value
            contact = np.flatnonzero(ref.gap <= tol)
            for k, (_, kind, param) in enumerate(rules):
                a_hat = predict(kind, param, red, g_tilde, problem, mu, snap)
                out["recall"][k, i, j], out["oversnap"][k, i, j] = recall_and_oversnap(a_hat, ref.strictly_active,
                                                                                       contact)
                out["size"][k, i, j] = a_hat.size
                for name in SNAPPED:
                    cert = ev.certify_correction(name, mu, K, f, c, ref, red, a_hat)
                    for field in FIELDS:
                        out[f"{field}_{name}"][k, i, j] = getattr(cert, field)
        LOG.info("r=%d done (%d queries)", r, n_q)
    add_best(out, n_rule)
    return out


def add_best(out: dict, n_rule: int) -> None:
    """Best of (R0, R0s, Bs) per rule, query and basis size."""
    acc, err, delta = (np.full_like(out["error_R0s"], np.nan) for _ in range(3))
    for k in range(n_rule):
        view = {f"{f}_{m}": out[f"{f}_{m}"][k] for m in SNAPPED for f in FIELDS}
        view.update({f"{f}_R0": out[f"{f}_R0"] for f in FIELDS})
        best = best_of(view, ("R0", "R0s", "Bs"))
        acc[k], err[k], delta[k] = best["accepted"], best["error"], best["delta"]
    out["accepted_best"], out["error_best"], out["delta_best"] = acc, err, delta


def summarise(out: dict) -> None:
    last, norm = -1, out["norm_u"]
    LOG.info("r=%d | R0: acc %.0f %%, eff %.2f, |M| %d, ||lam_M||_1 %.3g | B: acc %.0f %%, eff %.2f, |M| %d, "
             "||lam_M||_1 %.3g (medians)", out["r_values"][last],
             100 * np.nanmean(out["accepted_R0"][last]), np.nanmedian(out["delta_R0"][last] / out["error_R0"][last]),
             np.nanmedian(out["n_missed_R0"][last]), np.nanmedian(out["lam_missed_R0"][last]),
             100 * np.nanmean(out["accepted_B"][last]), np.nanmedian(out["delta_B"][last] / out["error_B"][last]),
             np.nanmedian(out["n_missed_B"][last]), np.nanmedian(out["lam_missed_B"][last]))
    for k, label in enumerate(out["rule_labels"]):
        LOG.info("%-14s recall %.2f over-snap %.2f |A^| %3d | R0s err %.2f %% eff %.2f | Bs err %.2f %% eff %.2f | "
                 "accepted best %.0f %%", label, np.nanmedian(out["recall"][k, last]),
                 np.nanmedian(out["oversnap"][k, last]), np.nanmedian(out["size"][k, last]),
                 100 * np.nanmedian(out["error_R0s"][k, last] / norm),
                 np.nanmedian(out["delta_R0s"][k, last] / out["error_R0s"][k, last]),
                 100 * np.nanmedian(out["error_Bs"][k, last] / norm),
                 np.nanmedian(out["delta_Bs"][k, last] / out["error_Bs"][k, last]),
                 100 * np.nanmean(out["accepted_best"][k, last]))


def main(args) -> int:
    problem = make_problem(args.problem)
    settings = SETTINGS[args.problem]
    path = result_path(EXPERIMENT, args.problem, args.seed)
    LOG.info("%s: %d rules x %d basis sizes -> %s", problem.name, len(rule_list()), N_SIZES, path)
    if args.dry_run:
        return cli.EXIT_OK
    out = run(problem, settings, args.seed)
    summarise(out)
    save_result(path, out, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed, "tau": TAU,
                            "rules": [r[0] for r in rule_list()], "settings": settings,
                            "inner_product": "X = K(mu_bar)", "multiplier": "S1", "projection_basis": "compliance"})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
