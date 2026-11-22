"""Numerical checks of ANALYSIS_T1_T3.md on the X1 queries (offline; uses the full-order multiplier).

Per test query and primal basis size, for R0 and for R0s with the gap rule (t = tau_gap * char_gap):
  T1  missed rows M, m(M) = ||B^T lam_M||_{K^-1}, the excess lam . g^, its bound m(M) min(||e||, ||e~||), the S1
      certificate and the bound (T1.2) with s^2 = gamma_X / alpha_X; position relative to the crossover m(M) / s^2.
  T2  accuracy change of snapping, ||B^+ p||_K and beta ||p||_2, for the gap rule and the multiplier rule.
  T3  S1-restricted on S_t = {g^ <= t} for R0s: coverage of A_+, certificate relative to S1, support size.
Output: results/theory/<problem>_seed<seed>.npz and a summary in the log.

Usage:  python experiments/check_theory.py --problem membrane --seed 0
"""

import logging

import numpy as np
from scipy.sparse.linalg import factorized

from certcorrect import cli
from certcorrect.active_set import by_gap, by_multiplier
from certcorrect.analysis import loglog_slope
from certcorrect.certificate import certify
from certcorrect.config import SEEDS, SETTINGS, TAU_LAMBDA, make_problem, result_path
from certcorrect.corrections import RightInverse, repair_r0, repair_r0_snap
from certcorrect.fom import energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.pipeline import EnergyInnerProduct, build_bases, rom_for_size, solve_many, training_snapshots
from certcorrect.results_io import save_result
from certcorrect.theory import contact_intensity, missed_rows, right_inverse_energy_norm, snap_difference

LOG = logging.getLogger("check_theory")
EXPERIMENT = "theory"
TAU_GAP = 1e-3
REL = 1e-9                         # round-off allowance when counting bound violations
TWO_OVER_ROOT3 = 2.0 / np.sqrt(3.0)
N_TEST_MAX = 100                   # queries per basis size (subset of the X1 test set)
CORRECTIONS = ("R0", "R0s")


def gamma_alpha(problem, inner, mu) -> tuple[float, float]:
    """``(gamma_X, alpha_X)`` of ``K(mu)`` relative to ``X = K(mu_bar)`` (max / min theta ratio)."""
    ratio = problem.theta(mu) / inner.theta_bar
    return float(ratio.max()), float(ratio.min())


def evaluate(problem, inner, s1, Bp, beta, mu, ref, red) -> dict:
    B = problem.B
    K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
    solve_k = factorized(K.tocsc()) if problem.k_varies else inner.solve
    gamma, alpha = gamma_alpha(problem, inner, mu)
    t = TAU_GAP * problem.char_gap
    g_tilde = B @ red.u - c
    e_tilde = energy_norm(K, ref.u - red.u)
    u_r0 = repair_r0(red.u, B, c, Bp)
    sets = {"R0": np.zeros(0, dtype=int), "R0s": by_gap(g_tilde, TAU_GAP, problem.char_gap)}
    row = {"e_tilde": e_tilde, "s2": gamma / alpha}
    for name, a_hat in sets.items():
        u_hat = repair_r0_snap(red.u, B, c, Bp, a_hat)
        g_hat = np.maximum(B @ u_hat - c, 0.0)
        e = energy_norm(K, ref.u - u_hat)
        M = missed_rows(ref.lam, g_tilde, a_hat)
        m = contact_intensity(B, ref.lam, M, solve_k)
        lam_s1, _ = s1.solve(f - K @ u_hat, g_hat, alpha)
        cert = certify(K, f, B, c, u_hat, lam_s1, inner.solve, alpha)
        row.update({f"lamM_l1_{name}": float(ref.lam[M].sum()), f"lamM_l2_{name}": float(np.linalg.norm(ref.lam[M])),
                    f"ghatM_l2_{name}": float(np.linalg.norm(g_hat[M])),
                    f"ghatM_max_{name}": float(g_hat[M].max(initial=0.0))})
        row.update({f"e_{name}": e, f"n_missed_{name}": M.size, f"m_{name}": m,
                    f"excess_{name}": float(ref.lam @ (B @ u_hat - c)), f"excess_bound_{name}": m * min(e, e_tilde),
                    f"delta_{name}": cert.delta,
                    f"t1_bound_{name}": TWO_OVER_ROOT3 * np.sqrt(gamma / alpha * e**2 + m * min(e, e_tilde))})
        if name == "R0s":                                   # T3: support S_t = {g^ <= t}
            support = np.flatnonzero(g_hat <= t)
            lam_r, _ = s1.solve(f - K @ u_hat, g_hat, alpha, support)
            cert_r = certify(K, f, B, c, u_hat, lam_r, inner.solve, alpha)
            row.update({"t3_covers": bool(np.isin(ref.strictly_active, support).all()),
                        "t3_support": support.size, "t3_ratio": cert_r.delta / cert.delta})
    e_r0 = energy_norm(K, ref.u - u_r0)
    for rule, a_hat in (("gap", sets["R0s"]), ("multiplier", by_multiplier(red.lam, TAU_LAMBDA))):
        p = snap_difference(g_tilde, a_hat)
        u_s = repair_r0_snap(red.u, B, c, Bp, a_hat)
        row.update({f"t2_change_{rule}": energy_norm(K, ref.u - u_s) - e_r0,
                    f"t2_exact_{rule}": energy_norm(K, Bp.apply(p)), f"t2_bound_{rule}": beta * np.linalg.norm(p)})
    row["t2_apriori_gap"] = beta * t * np.sqrt(len(sets["R0s"]))
    return row


def run(problem, settings: dict, seed: int) -> dict[str, np.ndarray]:
    snap = training_snapshots(problem, np.random.default_rng(SEEDS["train"] + seed), settings["n_train"])
    V, W = build_bases(snap, max(settings["r_primal"]), settings["r_dual"], settings["dual_kind"])
    mus = problem.sample(np.random.default_rng(SEEDS["test"] + seed), settings["n_test"])[:N_TEST_MAX]
    refs, ok = solve_many(problem, mus, "test")
    mus = mus[ok]
    inner = EnergyInnerProduct(problem)
    s1 = MinResidualS1(problem.B, inner.solve)
    Bp = RightInverse.from_constraints(problem.B)
    beta = right_inverse_energy_norm(problem.stiffness(inner.mu_bar), Bp)
    rows, r_of_row = [], []
    for r in settings["r_primal"]:
        if r > V.shape[1]:
            continue
        rom = rom_for_size(problem, V, W, r)
        for mu, ref in zip(mus, refs):
            red = rom.solve(mu)
            if red.converged:
                rows.append(evaluate(problem, inner, s1, Bp, beta, mu, ref, red))
                r_of_row.append(r)
    out = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    out["r"] = np.array(r_of_row)
    out["beta"] = np.array(beta)
    return out


def summarise(out: dict) -> None:
    for name in CORRECTIONS:
        e, m, s2 = out[f"e_{name}"], out[f"m_{name}"], out["s2"]
        viol_excess = np.mean(out[f"excess_{name}"] > out[f"excess_bound_{name}"] * (1 + REL) + 1e-14)
        viol_t1 = np.mean(out[f"delta_{name}"] > out[f"t1_bound_{name}"] * (1 + REL))
        missed = out[f"n_missed_{name}"] > 0
        pre = e[missed] >= m[missed] / s2[missed]
        eff = out[f"delta_{name}"] / e
        LOG.info("%-3s T1: bound violations excess %.1f %%, certificate %.1f %% | queries with missed rows %.0f %% "
                 "(median %d rows) | of those, pre-asymptotic (||e|| >= m(M)/s^2) %.0f %% | median m(M)/||e|| %.3f",
                 name, 100 * viol_excess, 100 * viol_t1, 100 * missed.mean(),
                 np.median(out[f"n_missed_{name}"][missed]) if missed.any() else 0, 100 * pre.mean() if missed.any()
                 else float("nan"), np.median(m[missed] / e[missed]) if missed.any() else float("nan"))
        if missed.any():
            ex = out[f"excess_{name}"][missed]
            local = out[f"lamM_l2_{name}"][missed] * out[f"ghatM_l2_{name}"][missed]
            LOG.info("      tightness (excess / bound), median: energy form (T1.1) %.3f | localized ||lam_M|| ||g^_M|| "
                     "%.3f | excess ~ ||e||^%.2f, ||lam_M||_1 ~ ||e||^%.2f, max g^_M ~ ||e||^%.2f",
                     np.median(ex / out[f"excess_bound_{name}"][missed]), np.median(ex / local),
                     loglog_slope(e[missed], ex), loglog_slope(e[missed], out[f"lamM_l1_{name}"][missed]),
                     loglog_slope(e[missed], out[f"ghatM_max_{name}"][missed]))
            ratio = m[missed] / e[missed]
            bins = np.quantile(ratio, [0, 1 / 3, 2 / 3, 1])
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (ratio >= lo) & (ratio <= hi)
                LOG.info("      m(M)/||e|| in [%.3g, %.3g]: median eff %.2f (T1.3 bound %.2f)", lo, hi,
                         np.median(eff[missed][sel]), TWO_OVER_ROOT3 * (1 + np.sqrt(hi)))
    for rule in ("gap", "multiplier"):
        viol = np.mean(np.abs(out[f"t2_change_{rule}"]) > out[f"t2_exact_{rule}"] * (1 + REL) + 1e-14)
        rel = out[f"t2_change_{rule}"] / out["e_R0"]
        LOG.info("T2 %-10s: bound violations %.1f %% | median relative change of the error %+.1f %% | median "
                 "||B^+p||_K / beta||p|| %.2f", rule, 100 * viol, 100 * np.median(rel),
                 np.median(out[f"t2_exact_{rule}"] / np.maximum(out[f"t2_bound_{rule}"], 1e-300)))
    LOG.info("T2 gap a priori bound (T2.3) holds in %.1f %% | median bound / actual cost %.1f",
             100 * np.mean(np.abs(out["t2_change_gap"]) <= out["t2_apriori_gap"] * (1 + REL)),
             np.median(out["t2_apriori_gap"] / np.maximum(np.abs(out["t2_change_gap"]), 1e-300)))
    cov = out["t3_covers"]
    LOG.info("T3 S_t: covers A_+ in %.0f %% | Delta(S_t)/Delta(S1) median %.3f (covered), %.3f (not covered) | "
             "median |S_t| %d", 100 * cov.mean(), np.median(out["t3_ratio"][cov]) if cov.any() else np.nan,
             np.median(out["t3_ratio"][~cov]) if (~cov).any() else np.nan, np.median(out["t3_support"]))


def main(args) -> int:
    problem = make_problem(args.problem)
    settings = SETTINGS[args.problem]
    path = result_path(EXPERIMENT, args.problem, args.seed)
    LOG.info("%s: up to %d queries x %s basis sizes -> %s", problem.name, N_TEST_MAX, settings["r_primal"], path)
    if args.dry_run:
        return cli.EXIT_OK
    out = run(problem, settings, args.seed)
    summarise(out)
    save_result(path, out, {"experiment": EXPERIMENT, "problem": args.problem, "seed": args.seed, "tau_gap": TAU_GAP,
                            "settings": settings})
    return cli.EXIT_OK


if __name__ == "__main__":
    cli.run(main, cli.base_parser(__doc__.splitlines()[0]))
