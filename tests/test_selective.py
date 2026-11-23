"""Selective enrichment: working-set QP, full selection = G, online certificate = full-order certificate."""

import numpy as np
import pytest

from certcorrect.bases import dual_basis, pod_basis
from certcorrect.certificate import certify
from certcorrect.corrections import EnergyProjection, RightInverse
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.offline_online import OnlineCertifier
from certcorrect.pipeline import EnergyInnerProduct
from certcorrect.problems import ElasticBlock
from certcorrect.selective import RawEnrichment, select_random, select_training_force, solve_working_set

TAU, TAU_GAP = 0.05, 3e-3


@pytest.fixture(scope="module")
def setup():
    problem = ElasticBlock(n_xy=6, nz=2, layered=True)
    rng = np.random.default_rng(0)
    sols = [FullOrderSolver(problem).solve(mu) for mu in problem.sample(rng, 15)]
    L = np.column_stack([s.lam for s in sols])
    V = pod_basis(np.column_stack([s.u for s in sols]), 6)
    W = dual_basis("nmf", L, 6)
    inner = EnergyInnerProduct(problem)
    Bp = RightInverse.from_constraints(problem.B)
    s1 = MinResidualS1(problem.B, inner.solve)
    raw = RawEnrichment(problem, V, s1.xinv_bt, Bp)
    online = OnlineCertifier(problem, V, W, inner, Bp, projection=raw)
    proj = EnergyProjection(problem, V, inner.solve, inner.matrix, Bp, "compliance", xinv_bt=s1.xinv_bt)
    online_g = OnlineCertifier(problem, V, W, inner, Bp, projection=proj)
    return problem, L, inner, s1, online, online_g, problem.sample(rng, 3)


def test_working_set_qp_kkt():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    H, b = A @ A.T + np.eye(5), rng.standard_normal(5)
    C, c = rng.standard_normal((8, 5)), rng.standard_normal(8) - 1.0
    q, feasible, _ = solve_working_set(H, b, C, c)
    assert feasible and (C @ q - c).min() >= -1e-9
    # KKT: H q - b = C^T nu with nu >= 0 supported on active rows (least squares on the active set)
    act = np.flatnonzero(np.abs(C @ q - c) < 1e-8)
    nu = np.linalg.lstsq(C[act].T, H @ q - b, rcond=None)[0] if act.size else np.zeros(0)
    assert np.allclose(C[act].T @ nu, H @ q - b, atol=1e-8) and (nu >= -1e-8).all()


def test_full_selection_equals_g(setup):
    problem, _, _, _, online, online_g, mus = setup
    for mu in mus:
        full = online.evaluate(mu, TAU, TAU_GAP, correction="GS", select=np.arange(problem.n_contact))
        g = online_g.evaluate(mu, TAU, TAU_GAP, correction="G")
        u_full, u_g = online.displacement(full), online_g.displacement(g)
        assert full.feasible
        assert np.abs(u_full - u_g).max() <= 1e-7 * np.abs(u_g).max()


def test_selective_certificate_matches_full_order(setup):
    problem, L, inner, s1, online, _, mus = setup
    for S in (select_random(problem.n_contact, 5, 0), select_training_force(L, 5), "predicted"):
        for mu in mus:
            res = online.evaluate(mu, TAU, TAU_GAP, correction="GS", select=S, predicted_factor=10)
            u_hat = online.displacement(res)
            K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
            assert (problem.B @ u_hat - c).min() >= -1e-12
            alpha = inner.alpha(mu)
            lam, _ = s1.solve(f - K @ u_hat, np.maximum(problem.B @ u_hat - c, 0.0), alpha)
            ref = certify(K, f, problem.B, c, u_hat, lam, inner.solve, alpha)
            assert res.delta == pytest.approx(ref.delta, rel=1e-6, abs=1e-7 * energy_norm(K, u_hat))
