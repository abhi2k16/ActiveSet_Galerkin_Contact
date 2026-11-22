"""Checks of ANALYSIS_T1_T3.md: Proposition E2 example, T1 identity and bound, T2 identity, T3 bound and floor."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from certcorrect.certificate import certify
from certcorrect.corrections import RightInverse, repair_r0, repair_r0_snap
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.problems import Membrane
from certcorrect.theory import (contact_intensity, missed_rows, right_inverse_energy_norm, snap_difference,
                                support_floor)

REL = 1e-9                       # relative round-off allowance in the identities
TWO_OVER_ROOT3 = 2.0 / np.sqrt(3.0)
NOISE = 0.01                     # perturbation of the exact solution, mimicking a ROM error


def s1_certificate(K, f, B, c, u_hat, solve_k, s1, support=None):
    g_hat = np.maximum(B @ u_hat - c, 0.0)
    lam, ok = s1.solve(f - K @ u_hat, g_hat, 1.0, support)
    assert ok
    return certify(K, f, B, c, u_hat, lam, solve_k, 1.0)


@pytest.mark.parametrize("eps", [0.5, 0.1, 1e-2, 1e-4])
def test_proposition_e2_example_and_t1_sharpness(eps):
    """Paper Prop. 3.3: N = 2, n_c = 1, K = I, B = [1 0], c = 0, f = (-1, 0). The optimal certificate over all
    multipliers is sqrt(eps(1 + eps)) at l = 1 + eps; S1 lies between it and the bound (2/sqrt3) sqrt(eps^2 + eps)."""
    K, B = sp.identity(2, format="csr"), sp.csr_matrix([[1.0, 0.0]])
    f, c, u_hat = np.array([-1.0, 0.0]), np.array([0.0]), np.array([eps, 0.0])
    solve = lambda r: np.array(r, dtype=float)             # noqa: E731  (X = K = I)
    optimum = np.sqrt(eps * (1 + eps))
    assert certify(K, f, B, c, u_hat, np.array([1 + eps]), solve, 1.0).delta == pytest.approx(optimum, rel=1e-12)
    grid = np.linspace(0.0, 4.0, 4001)
    assert min(certify(K, f, B, c, u_hat, np.array([t]), solve, 1.0).delta for t in grid) >= optimum * (1 - 1e-12)
    cert = s1_certificate(K, f, B, c, u_hat, solve, MinResidualS1(B, solve))
    assert optimum * (1 - REL) <= cert.delta <= TWO_OVER_ROOT3 * np.sqrt(eps**2 + eps) * (1 + REL)


@pytest.fixture(scope="module")
def case():
    problem = Membrane(n=15)
    mu = problem.sample(np.random.default_rng(0), 1)[0]
    K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
    sol = FullOrderSolver(problem).solve(mu, K=K)
    solve_k = factorized(K.tocsc())
    u_tilde = sol.u + NOISE * np.random.default_rng(1).standard_normal(problem.n_dof)
    return dict(problem=problem, K=K, f=f, c=c, sol=sol, solve_k=solve_k, u_tilde=u_tilde,
                Bp=RightInverse.from_constraints(problem.B), s1=MinResidualS1(problem.B, solve_k))


def active_sets(case):
    """R0 (empty), a random half of A_+, and A_+ itself."""
    plus = case["sol"].strictly_active
    half = np.random.default_rng(2).choice(plus, size=len(plus) // 2, replace=False)
    return {"R0": np.zeros(0, dtype=int), "half": half, "full": plus}


def test_t1_identity_and_bound(case):
    K, f, c, sol, B = case["K"], case["f"], case["c"], case["sol"], case["problem"].B
    g_tilde = B @ case["u_tilde"] - c
    e_tilde = sol.u - case["u_tilde"]
    seen_missed = False
    for name, a_hat in active_sets(case).items():
        u_hat = repair_r0_snap(case["u_tilde"], B, c, case["Bp"], a_hat)
        e = sol.u - u_hat
        g_hat = B @ u_hat - c
        M = missed_rows(sol.lam, g_tilde, a_hat)
        seen_missed |= M.size > 0
        excess = float(sol.lam @ g_hat)
        lam_M = np.zeros_like(sol.lam)
        lam_M[M] = sol.lam[M]
        scale = max(abs(excess), 1e-12)
        assert excess == pytest.approx(-lam_M @ (B @ e_tilde), rel=REL, abs=REL * scale), name
        assert excess == pytest.approx(-lam_M @ (B @ e), rel=REL, abs=REL * scale), name
        m = contact_intensity(B, sol.lam, M, case["solve_k"])
        e_norm, et_norm = energy_norm(K, e), energy_norm(K, e_tilde)
        assert excess <= m * min(e_norm, et_norm) * (1 + REL), name
        cert = s1_certificate(K, f, B, c, u_hat, case["solve_k"], case["s1"])
        assert cert.delta <= TWO_OVER_ROOT3 * np.sqrt(e_norm**2 + m * min(e_norm, et_norm)) * (1 + REL), name
        if name == "full":
            assert M.size == 0 and cert.delta <= TWO_OVER_ROOT3 * e_norm * (1 + REL)
    assert seen_missed                                       # the perturbation does create missed rows
    np.testing.assert_allclose(repair_r0(case["u_tilde"], B, c, case["Bp"]),
                               repair_r0_snap(case["u_tilde"], B, c, case["Bp"], np.zeros(0, dtype=int)))


def test_t2_identity_and_bound(case):
    K, c, sol, B, Bp = case["K"], case["c"], case["sol"], case["problem"].B, case["Bp"]
    g_tilde = B @ case["u_tilde"] - c
    u_r0 = repair_r0(case["u_tilde"], B, c, Bp)
    beta = right_inverse_energy_norm(K, Bp)
    dense = np.linalg.eigvalsh((B @ K @ B.T).toarray())[-1]           # unit selection: (B^+)^T K B^+ = B K B^T
    assert beta**2 == pytest.approx(dense, rel=1e-6)
    a_hat = np.flatnonzero(g_tilde <= 0.05)                            # gap rule, t = 0.05
    snap = snap_difference(g_tilde, a_hat)
    u_s = repair_r0_snap(case["u_tilde"], B, c, Bp, a_hat)
    np.testing.assert_allclose(u_s, u_r0 - Bp.apply(snap), atol=1e-13)
    gap = abs(energy_norm(K, sol.u - u_s) - energy_norm(K, sol.u - u_r0))
    assert gap <= energy_norm(K, Bp.apply(snap)) * (1 + REL) <= beta * np.linalg.norm(snap) * (1 + 1e-6)
    assert snap.max() <= 0.05 and beta * np.linalg.norm(snap) <= beta * 0.05 * np.sqrt(len(a_hat)) * (1 + REL)


def test_t3_support_bound_and_floor(case):
    K, f, c, sol, s1, solve_k = (case[k] for k in ("K", "f", "c", "sol", "s1", "solve_k"))
    B = case["problem"].B
    plus = sol.strictly_active
    # (a) S covering A_+: S1-restricted obeys the bound (E1) of S1, here at u^ = R0 repair.
    u_hat = repair_r0(case["u_tilde"], B, c, case["Bp"])
    e_norm = energy_norm(K, sol.u - u_hat)
    excess = float(sol.lam @ (B @ u_hat - c))
    cert = s1_certificate(K, f, B, c, u_hat, solve_k, s1, support=plus)
    assert cert.delta <= TWO_OVER_ROOT3 * np.sqrt(e_norm**2 + excess) * (1 + REL)
    # (b) S missing the row with the largest force: floor sigma_S > 0 even at the exact solution.
    support = np.setdiff1d(plus, [plus[np.argmax(sol.lam[plus])]])
    sigma = support_floor(B, sol.lam, support, solve_k)
    exact_restricted = s1_certificate(K, f, B, c, sol.u, solve_k, s1, support=support)
    exact_full = s1_certificate(K, f, B, c, sol.u, solve_k, s1)
    assert sigma > 0 and exact_restricted.delta >= sigma * (1 - REL)
    assert exact_full.delta <= 1e-6 * exact_restricted.delta
    # (c) T3.1 with the missed intensity m(S^c).
    m_out = contact_intensity(B, sol.lam, np.setdiff1d(plus, support), solve_k)
    cert = s1_certificate(K, f, B, c, u_hat, solve_k, s1, support=support)
    assert cert.delta <= TWO_OVER_ROOT3 * np.sqrt((e_norm + m_out) ** 2 + excess) * (1 + REL)
    assert cert.delta >= (sigma - e_norm) * (1 - REL)
