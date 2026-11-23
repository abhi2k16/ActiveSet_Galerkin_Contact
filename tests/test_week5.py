"""Week 5: energy projections B / Bs, mixed LCP, S0 / S2, neighbours rule, support rule, lift-off benchmark."""

import numpy as np
import pytest
from scipy.sparse.linalg import factorized

from certcorrect.active_set import by_neighbours, support_by_gap
from certcorrect.bases import pod_basis
from certcorrect.config import make_problem
from certcorrect.corrections import EnergyProjection, RightInverse, solve_mixed_lcp
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1, clip_s0, nnls_s2
from certcorrect.problems import Membrane
from certcorrect.rom import solve_complementarity

FEAS_TOL = 1e-12
NOISE = 0.02
N_SNAP = 15


@pytest.fixture(scope="module")
def setup():
    problem = Membrane(n=15)
    rng = np.random.default_rng(0)
    solver = FullOrderSolver(problem)
    sols = [solver.solve(mu) for mu in problem.sample(rng, N_SNAP)]
    V = pod_basis(np.column_stack([s.u for s in sols]), 6)
    K = problem.stiffness(problem.lo)                      # constant K: X = K
    solve_k = factorized(K.tocsc())
    Bp = RightInverse.from_constraints(problem.B)
    s1 = MinResidualS1(problem.B, solve_k)
    mu = problem.sample(rng, 1)[0]
    ref = solver.solve(mu)
    u_tilde = V @ (V.T @ ref.u) + NOISE * (V @ rng.standard_normal(V.shape[1]))   # a prediction in range V
    proj = {b: EnergyProjection(problem, V, solve_k, K, Bp, b, xinv_bt=s1.xinv_bt) for b in ("compliance", "nodal")}
    return dict(problem=problem, K=K, solve_k=solve_k, s1=s1, mu=mu, ref=ref, u_tilde=u_tilde, proj=proj, V=V,
                c=problem.offset(mu))


def test_compliance_projection_is_the_full_space_energy_projection(setup):
    """For X = K the full-space K-projection is u~ + K^-1 B^T nu with the Delassus LCP; B must reproduce it."""
    p, c, u_t, s1 = setup["problem"], setup["c"], setup["u_tilde"], setup["s1"]
    nu, ok = solve_complementarity(s1.S, p.B @ u_t - c)
    assert ok
    reference = u_t + s1.xinv_bt @ nu
    res = setup["proj"]["compliance"].project(setup["mu"], u_t, c)
    assert res.ok and (p.B @ res.u - c).min() >= -FEAS_TOL
    K = setup["K"]
    assert energy_norm(K, res.u - reference) <= 1e-8 * energy_norm(K, reference)
    assert res.repair_norm <= 1e-8 * res.projection_norm + 1e-14


def test_feasible_prediction_is_unchanged(setup):
    p, c = setup["problem"], setup["c"]
    feasible = setup["ref"].u                                  # exact solution: feasible (and in a larger space)
    proj = EnergyProjection(p, np.column_stack([setup["V"], feasible]), setup["solve_k"], setup["K"],
                            RightInverse.from_constraints(p.B), "compliance", xinv_bt=setup["s1"].xinv_bt)
    res = proj.project(setup["mu"], feasible, c)
    assert energy_norm(setup["K"], res.u - feasible) <= 1e-9 * energy_norm(setup["K"], feasible)


def test_snapped_projection_and_nodal_basis(setup):
    p, c, u_t, K = setup["problem"], setup["c"], setup["u_tilde"], setup["K"]
    a_hat = setup["ref"].strictly_active
    for basis, proj in setup["proj"].items():
        res = proj.project(setup["mu"], u_t, c, a_hat)
        gap = p.B @ res.u - c
        assert res.ok and np.abs(gap[a_hat]).max() <= FEAS_TOL and gap.min() >= -FEAS_TOL, basis
    # The compliance projection is the global K-projection; the nodal one cannot be closer to u~.
    d_comp = energy_norm(K, setup["proj"]["compliance"].project(setup["mu"], u_t, c).u - u_t)
    d_nodal = energy_norm(K, setup["proj"]["nodal"].project(setup["mu"], u_t, c).u - u_t)
    assert d_comp <= d_nodal * (1 + 1e-9)


def test_mixed_lcp():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((6, 6))
    M = A @ A.T + 6 * np.eye(6)
    q = rng.standard_normal(6)
    equal = np.array([1, 4])
    nu, ok = solve_mixed_lcp(M, q, equal)
    w = M @ nu + q
    rest = np.setdiff1d(np.arange(6), equal)
    assert ok and np.abs(w[equal]).max() < 1e-10
    assert nu[rest].min() >= -1e-12 and w[rest].min() >= -1e-10 and abs(nu[rest] @ w[rest]) < 1e-10


def test_s0_s2(setup):
    p, K, c, s1, solve_k = setup["problem"], setup["K"], setup["c"], setup["s1"], setup["solve_k"]
    np.testing.assert_array_equal(clip_s0(np.array([-1.0, 0.0, 2.0])), [0.0, 0.0, 2.0])
    u_hat = setup["proj"]["compliance"].project(setup["mu"], setup["u_tilde"], c).u
    a = p.load(setup["mu"]) - K @ u_hat
    lam, ok = nnls_s2(s1, a, 1.0)
    grad = p.B @ solve_k(a + p.B.T @ lam)                     # optimality of the residual-only problem
    scale = np.abs(grad).max()
    assert ok and lam.min() >= 0 and grad.min() >= -1e-8 * scale and np.abs(lam * grad).max() <= 1e-8 * scale


def test_neighbours_and_support_rules():
    mus = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    act = [np.array([1, 2]), np.array([5]), np.array([2, 7])]
    lo, hi = np.zeros(2), np.array([1.0, 10.0])               # second axis 10x wider: scaled distance
    np.testing.assert_array_equal(by_neighbours(np.array([0.1, 0.0]), mus, act, 1, lo, hi), [1, 2])
    np.testing.assert_array_equal(by_neighbours(np.array([0.1, 0.0]), mus, act, 2, lo, hi), [1, 2, 7])
    with pytest.raises(ValueError):
        by_neighbours(np.zeros(2), mus, act, 0, lo, hi)
    np.testing.assert_array_equal(support_by_gap(np.array([0.0, 0.2, 0.05]), 0.1), [0, 2])


def test_liftoff_benchmark_sampling():
    problem = make_problem("membrane_shift")
    rng = np.random.default_rng(4)
    train, test = problem.sample(rng, 50), problem.sample_test(rng, 50)
    assert train[:, 1].max() <= 0.45 and test[:, 1].min() >= 0.55
    assert problem.in_box(train).all() and not problem.in_box(test).any()


def test_enriched_galerkin_baseline(setup):
    """G is feasible, satisfies the reduced KKT conditions, and is exact when u lies in range Z."""
    from certcorrect.baselines.enriched_galerkin import EnrichedGalerkin

    p, K, c, mu = setup["problem"], setup["K"], setup["c"], setup["mu"]
    res = EnrichedGalerkin(setup["proj"]["compliance"]).solve(mu)
    Z = setup["proj"]["compliance"].Z
    assert res.ok and (p.B @ res.u - c).min() >= -FEAS_TOL and res.nu.min() >= -1e-12
    stationarity = Z.T @ (K @ (Z @ res.q) - p.load(mu)) - (p.B @ Z).T @ res.nu
    assert np.abs(stationarity).max() <= 1e-9 * np.abs(Z.T @ p.load(mu)).max()
    # Enrich V with K^-1 f(mu): for constant K and X = K the exact solution then lies in range Z.
    V_exact = np.column_stack([setup["V"], setup["solve_k"](p.load(mu))])
    proj = EnergyProjection(p, V_exact, setup["solve_k"], K, RightInverse.from_constraints(p.B), "compliance",
                            xinv_bt=setup["s1"].xinv_bt)
    exact = EnrichedGalerkin(proj).solve(mu)
    assert energy_norm(K, exact.u - setup["ref"].u) <= 1e-8 * energy_norm(K, setup["ref"].u)
