"""R0 / R0s (formulation Prop. 2), active-set rules, and S1 (Prop. 1, Lemma 2)."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from certcorrect.active_set import by_gap, by_multiplier, recall_and_oversnap
from certcorrect.certificate import certify
from certcorrect.corrections import RightInverse, is_selection, repair_r0, repair_r0_snap
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.problems import Membrane

LCP_TOL = 1e-8


@pytest.fixture(scope="module")
def membrane():
    problem = Membrane(n=15)
    mu = problem.sample(np.random.default_rng(0), 1)[0]
    K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
    sol = FullOrderSolver(problem).solve(mu, K=K)
    solve_k = factorized(K.tocsc())
    return problem, K, f, c, sol, solve_k, MinResidualS1(problem.B, solve_k)


def test_right_inverse_selection_and_general():
    rng = np.random.default_rng(0)
    sel = sp.csr_matrix((2.0 * np.ones(3), ([0, 1, 2], [4, 1, 7])), shape=(3, 9))
    gen = sp.csr_matrix(rng.standard_normal((3, 9)))
    assert is_selection(sel) and not is_selection(gen)
    for B in (sel, gen):
        Bp = RightInverse.from_constraints(B)
        d = rng.standard_normal(3)
        np.testing.assert_allclose(B @ Bp.apply(d), d, atol=1e-12)
        np.testing.assert_allclose(Bp.apply(d), np.linalg.pinv(B.toarray()) @ d, atol=1e-12)   # minimum norm


def test_rank_deficient_constraints_rejected():
    B = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0]]))
    with pytest.raises(ValueError):
        RightInverse.from_constraints(B)


def test_r0_and_r0_snap(membrane):
    problem, K, f, c, sol, _, _ = membrane
    rng = np.random.default_rng(1)
    u_tilde = sol.u + 0.05 * rng.standard_normal(problem.n_dof)
    B, Bp = problem.B, RightInverse.from_constraints(problem.B)
    bu = B @ u_tilde
    u0 = repair_r0(u_tilde, B, c, Bp)
    np.testing.assert_allclose(B @ u0, np.maximum(bu, c), atol=1e-14)          # Prop. 2(i)
    untouched = bu >= c
    np.testing.assert_array_equal((B @ u0)[untouched], bu[untouched])           # Prop. 2(ii)
    a_hat = sol.strictly_active
    us = repair_r0_snap(u_tilde, B, c, Bp, a_hat)
    gap = B @ us - c
    assert np.abs(gap[a_hat]).max() <= 1e-14 and gap.min() >= -1e-14


def test_active_set_rules():
    lam = np.array([0.0, 1e-12, 0.5, 1.0])
    np.testing.assert_array_equal(by_multiplier(lam, 1e-8), [2, 3])
    assert by_multiplier(-lam, 1e-8).size == 0
    np.testing.assert_array_equal(by_gap(np.array([-0.1, 0.0, 1e-9, 0.2]), 1e-8, 1.0), [0, 1, 2])
    recall, oversnap = recall_and_oversnap(np.array([1, 2]), np.array([1, 3]), np.array([1, 3]))
    assert recall == 0.5 and oversnap == 0.5


def test_s1_recovers_exact_multiplier(membrane):
    """Lemma 2: at u^ = u the S1 multiplier is lam and the certificate vanishes (X = K)."""
    problem, K, f, c, sol, solve_k, s1 = membrane
    g_hat = np.maximum(problem.B @ sol.u - c, 0.0)
    scale = np.abs(sol.lam).max()
    for support in (None, sol.strictly_active):
        lam, ok = s1.solve(f - K @ sol.u, g_hat, 1.0, support)
        assert ok and np.abs(lam - sol.lam).max() <= 1e-6 * scale
    cert = certify(K, f, problem.B, c, sol.u, lam, solve_k, 1.0)
    assert cert.delta <= 1e-6 * energy_norm(K, sol.u)


def test_s1_optimality_and_improvement(membrane):
    """Prop. 1: lam >= 0, grad Phi >= 0, lam . grad Phi = 0; and S1 is no worse than the true multiplier in Phi."""
    problem, K, f, c, sol, solve_k, s1 = membrane
    rng = np.random.default_rng(2)
    B, Bp = problem.B, RightInverse.from_constraints(problem.B)
    u_hat = repair_r0(sol.u + 0.05 * rng.standard_normal(problem.n_dof), B, c, Bp)
    g_hat, a = np.maximum(B @ u_hat - c, 0.0), f - K @ u_hat
    lam, ok = s1.solve(a, g_hat, 1.0)
    assert ok and lam.min() >= 0
    grad = 2.0 * (B @ solve_k(a + B.T @ lam)) + g_hat
    scale = np.abs(grad).max()
    assert grad.min() >= -LCP_TOL * scale and np.abs(lam * grad).max() <= LCP_TOL * scale * max(lam.max(), 1.0)

    def phi(l):
        r = a + B.T @ l
        return MinResidualS1.objective(float(r @ solve_k(r)), g_hat, l, 1.0)

    assert phi(lam) <= phi(sol.lam) * (1 + 1e-10)
