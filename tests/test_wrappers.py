"""Benchmarks, full-order solver, bases and Galerkin ROM (the fea_engine / rom_engine wrapper layer)."""

import numpy as np
import pytest

from certcorrect.bases import dual_basis, pod_basis
from certcorrect.fom import FullOrderSolver, energy_error, kkt_residuals
from certcorrect.problems import ElasticBlock, Membrane
from certcorrect.rom import GalerkinROM

KKT_TOL = 1e-8
N_TRAIN = 12

SMALL_PROBLEMS = {"membrane": lambda: Membrane(n=15), "block": lambda: ElasticBlock(n_xy=6, nz=2),
                  "layered": lambda: ElasticBlock(n_xy=6, nz=2, layered=True)}


@pytest.fixture(scope="module", params=sorted(SMALL_PROBLEMS))
def problem(request):
    return SMALL_PROBLEMS[request.param]()


def test_affine_stiffness_matches_terms(problem):
    mu = problem.sample(np.random.default_rng(0), 1)[0]
    th = problem.theta(mu)
    assert len(th) == len(problem.stiffness_terms) and np.all(th >= 0)
    K = problem.stiffness(mu)
    assert abs(K - K.T).max() < 1e-12
    assert problem.k_varies == (problem.name == "layered")


def test_outside_samples_leave_the_box(problem):
    rng = np.random.default_rng(3)
    for mode in ("load", "shift"):
        assert not problem.in_box(problem.sample_outside(rng, 20, mode)).any()
    with pytest.raises(ValueError):
        problem.sample_outside(rng, 1, "nowhere")


def test_full_order_solution_satisfies_kkt(problem):
    mu = problem.sample(np.random.default_rng(1), 1)[0]
    K, f = problem.stiffness(mu), problem.load(mu)
    sol = FullOrderSolver(problem).solve(mu, K=K)
    assert sol.converged and len(sol.strictly_active) > 0
    scale = max(np.abs(f).max(), 1.0)
    for name, value in kkt_residuals(K, f, problem.B, sol).items():
        assert value <= KKT_TOL * scale, name


def test_rom_reproduces_snapshot_with_ideal_basis(problem):
    """With V = span(u*) and W = span(lam*) the exact pair solves the one-dimensional reduced problem."""
    mu = problem.sample(np.random.default_rng(2), 1)[0]
    K = problem.stiffness(mu)
    sol = FullOrderSolver(problem).solve(mu, K=K)
    V = (sol.u / np.linalg.norm(sol.u))[:, None]
    W = (sol.lam / np.linalg.norm(sol.lam))[:, None]
    red = GalerkinROM(problem, V, W).solve(mu)
    assert red.converged and energy_error(K, sol.u, red.u) < 1e-8


def test_bases_from_snapshots(problem):
    solver, rng = FullOrderSolver(problem), np.random.default_rng(4)
    sols = [solver.solve(mu) for mu in problem.sample(rng, N_TRAIN)]
    U = np.column_stack([s.u for s in sols])
    L = np.column_stack([s.lam for s in sols])
    V = pod_basis(U, 5)
    np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-10)
    K0 = problem.stiffness(problem.lo)
    VK = pod_basis(U, 5, K=K0)
    np.testing.assert_allclose(VK.T @ (K0 @ VK), np.eye(VK.shape[1]), atol=1e-8)
    assert np.all(dual_basis("nmf", L, 4) >= 0)
    assert dual_basis("svd", L, 4).shape == (problem.n_contact, 4)
    with pytest.raises(ValueError):
        dual_basis("nmf", np.zeros_like(L), 4)
