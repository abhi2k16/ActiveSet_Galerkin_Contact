"""Feasible-pair certificate: exact solution gives zero (Lemma 2), bound holds on perturbed feasible pairs."""

import numpy as np
import pytest
from scipy.sparse.linalg import factorized

from certcorrect.certificate import certify, is_accepted, min_theta_alpha, quadratic_root_bound
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.problems import ElasticBlock, Membrane

SMALL_PROBLEMS = {"membrane": lambda: Membrane(n=15), "block": lambda: ElasticBlock(n_xy=6, nz=2)}
N_PERTURBATIONS = 5


@pytest.fixture(scope="module", params=sorted(SMALL_PROBLEMS))
def solved(request):
    problem = SMALL_PROBLEMS[request.param]()
    mu = problem.sample(np.random.default_rng(0), 1)[0]
    K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
    sol = FullOrderSolver(problem).solve(mu, K=K)
    assert sol.converged and len(sol.strictly_active) > 0
    return problem, K, f, c, sol, factorized(K.tocsc())          # X = K, alpha_X = 1


def test_exact_solution_has_zero_certificate(solved):
    problem, K, f, c, sol, solve_k = solved
    cert = certify(K, f, problem.B, c, sol.u, sol.lam, solve_k, 1.0)
    assert cert.is_admissible
    assert cert.delta <= 1e-6 * energy_norm(K, sol.u)


def test_certificate_bounds_error_of_feasible_perturbations(solved):
    problem, K, f, c, sol, solve_k = solved
    rng = np.random.default_rng(1)
    scale = np.abs(sol.u).max()
    for _ in range(N_PERTURBATIONS):
        # B selects unknowns with +1 entries, so a non-negative shift keeps B u^ >= c.
        u_hat = sol.u + 0.05 * scale * rng.random(problem.n_dof)
        lam_hat = np.maximum(sol.lam * (1.0 + 0.2 * rng.standard_normal(problem.n_contact)), 0.0)
        cert = certify(K, f, problem.B, c, u_hat, lam_hat, solve_k, 1.0)
        assert cert.is_admissible
        assert cert.delta >= energy_norm(K, sol.u - u_hat) * (1 - 1e-12)


def test_euclidean_certificate_bounds_error_and_is_looser(solved):
    """X = I with alpha = lambda_min(K): still a bound, and never sharper than X = K (Remark 1)."""
    from certcorrect.pipeline import EuclideanInnerProduct
    from certcorrect.verified import extreme_eigenvalues

    problem, K, f, c, sol, solve_k = solved
    eig = extreme_eigenvalues(K)
    dense = np.linalg.eigvalsh(K.toarray())
    assert eig.lambda_min == pytest.approx(dense[0], rel=1e-8) and eig.lambda_max == pytest.approx(dense[-1], rel=1e-8)
    inner = EuclideanInnerProduct(problem, dense[0] * (1 - 1e-9))     # safe side of the exact value
    rng = np.random.default_rng(5)
    u_hat = sol.u + 0.05 * np.abs(sol.u).max() * rng.random(problem.n_dof)
    error = energy_norm(K, sol.u - u_hat)
    cert_i = certify(K, f, problem.B, c, u_hat, sol.lam, inner.solve, inner.alpha(None))
    cert_k = certify(K, f, problem.B, c, u_hat, sol.lam, solve_k, 1.0)
    assert cert_i.delta >= error and cert_i.rho_bar >= cert_k.rho_bar * (1 - 1e-12)


def test_infeasible_pair_is_not_certified(solved):
    problem, K, f, c, sol, solve_k = solved
    cert = certify(K, f, problem.B, c, sol.u - 1.0, sol.lam, solve_k, 1.0)
    assert not cert.is_admissible and cert.delta == np.inf


def test_quadratic_root_and_acceptance():
    assert quadratic_root_bound(0.0, 0.0) == 0.0
    assert quadratic_root_bound(3.0, 0.0) == pytest.approx(3.0)
    assert quadratic_root_bound(0.0, 4.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        quadratic_root_bound(-1.0, 0.0)
    assert is_accepted(0.04, 1.0, 0.05) and not is_accepted(0.05, 1.0, 0.05)


def test_min_theta_alpha():
    assert min_theta_alpha([1.0, 2.0], [1.0, 1.0]) == 1.0
    assert min_theta_alpha([1.0, 0.5], [1.0, 1.0]) == 0.5
    with pytest.raises(ValueError):
        min_theta_alpha([1.0, -0.1], [1.0, 1.0])
