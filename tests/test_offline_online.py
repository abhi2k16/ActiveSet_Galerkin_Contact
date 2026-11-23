"""The offline-online certificate reproduces the full-order evaluation (affine and hybrid load)."""

import numpy as np
import pytest

from certcorrect.active_set import by_gap
from certcorrect.bases import dual_basis, pod_basis
from certcorrect.certificate import certify, is_accepted
from certcorrect.corrections import RightInverse, repair_r0_snap
from certcorrect.fom import FullOrderSolver, energy_norm
from certcorrect.multipliers import MinResidualS1
from certcorrect.offline_online import OnlineCertifier
from certcorrect.pipeline import EnergyInnerProduct
from certcorrect.problems import ElasticBlock, Membrane
from certcorrect.rom import GalerkinROM

N_SNAP, R, R_DUAL, N_QUERY = 15, 6, 6, 3
TAU, TAU_GAP = 0.05, 3e-3
CASES = {"layered (affine, Q=2)": lambda: ElasticBlock(n_xy=6, nz=2, layered=True),
         "block (affine)": lambda: ElasticBlock(n_xy=6, nz=2),
         "membrane (hybrid load)": lambda: Membrane(n=15)}


@pytest.fixture(scope="module", params=sorted(CASES))
def setup(request):
    problem = CASES[request.param]()
    rng = np.random.default_rng(0)
    sols = [FullOrderSolver(problem).solve(mu) for mu in problem.sample(rng, N_SNAP)]
    V = pod_basis(np.column_stack([s.u for s in sols]), R)
    W = dual_basis("nmf", np.column_stack([s.lam for s in sols]), R_DUAL)
    inner = EnergyInnerProduct(problem)
    Bp = RightInverse.from_constraints(problem.B)
    return problem, V, W, inner, Bp, OnlineCertifier(problem, V, W, inner, Bp), problem.sample(rng, N_QUERY)


def test_affine_load_matches(setup):
    problem = setup[0]
    affine = problem.load_affine()
    if affine is None:
        pytest.skip("non-affine load")
    terms, phi = affine
    mu = problem.sample(np.random.default_rng(5), 1)[0]
    np.testing.assert_allclose(sum(p * t for p, t in zip(phi(mu), terms)), problem.load(mu), atol=1e-14)


def test_online_equals_full_order(setup):
    problem, V, W, inner, Bp, online, mus = setup
    s1 = MinResidualS1(problem.B, inner.solve)
    rom = GalerkinROM(problem, V, W)
    for mu in mus:
        res = online.evaluate(mu, TAU, TAU_GAP)
        red = rom.solve(mu)
        np.testing.assert_allclose(res.u_r, red.u_r, rtol=1e-10, atol=1e-12)
        K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
        a_hat = by_gap(problem.B @ red.u - c, TAU_GAP, problem.char_gap)
        u_hat = repair_r0_snap(red.u, problem.B, c, Bp, a_hat)
        np.testing.assert_allclose(online.displacement(res), u_hat, atol=1e-12 * np.abs(u_hat).max())
        alpha = inner.alpha(mu)
        lam, _ = s1.solve(f - K @ u_hat, np.maximum(problem.B @ u_hat - c, 0.0), alpha)
        np.testing.assert_allclose(res.multiplier, lam, atol=1e-8 * max(np.abs(lam).max(), 1e-300))
        ref = certify(K, f, problem.B, c, u_hat, lam, inner.solve, alpha)
        assert res.delta == pytest.approx(ref.delta, rel=1e-7)
        assert res.u_norm == pytest.approx(energy_norm(K, u_hat), rel=1e-10)
        assert res.accepted == is_accepted(ref.delta, energy_norm(K, u_hat), TAU)


def test_online_r0_and_restricted_support(setup):
    problem, V, W, inner, Bp, online, mus = setup
    mu = mus[0]
    r0 = online.evaluate(mu, TAU, TAU_GAP, snap=False)
    full = online.evaluate(mu, TAU, TAU_GAP)
    restricted = online.evaluate(mu, TAU, TAU_GAP, support_factor=10)
    assert r0.g_hat.min() >= 0 and full.g_hat.min() >= 0
    phi = lambda res: res.rho_bar**2 + res.eta                       # noqa: E731  (S1 minimises Phi, not Delta)
    assert phi(restricted) >= phi(full) * (1 - 1e-9)
    assert np.isfinite(r0.delta) and r0.delta > 0


def test_online_bs_equals_full_order(setup):
    """Bs online (projection QP in coefficients + clean-up) reproduces EnergyProjection and the certificate."""
    from certcorrect.corrections import EnergyProjection

    problem, V, W, inner, Bp, _, mus = setup
    s1 = MinResidualS1(problem.B, inner.solve)
    proj = EnergyProjection(problem, V, inner.solve, inner.matrix, Bp, "compliance", xinv_bt=s1.xinv_bt)
    online = OnlineCertifier(problem, V, W, inner, Bp, projection=proj)
    rom = GalerkinROM(problem, V, W)
    for mu in mus:
        res = online.evaluate(mu, TAU, TAU_GAP, correction="Bs")
        red = rom.solve(mu)
        K, f, c = problem.stiffness(mu), problem.load(mu), problem.offset(mu)
        a_hat = by_gap(problem.B @ red.u - c, TAU_GAP, problem.char_gap)
        u_hat = proj.project(mu, red.u, c, a_hat).u
        scale = np.abs(u_hat).max()
        np.testing.assert_allclose(online.displacement(res), u_hat, atol=1e-9 * scale)
        alpha = inner.alpha(mu)
        lam, _ = s1.solve(f - K @ u_hat, np.maximum(problem.B @ u_hat - c, 0.0), alpha)
        ref = certify(K, f, problem.B, c, u_hat, lam, inner.solve, alpha)
        assert res.delta == pytest.approx(ref.delta, rel=1e-6)
        assert res.u_norm == pytest.approx(energy_norm(K, u_hat), rel=1e-9)
