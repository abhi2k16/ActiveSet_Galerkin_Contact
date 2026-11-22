"""Steps shared by the experiments: training snapshots, bases, the inner product ``X``, and the evaluation of one
query (G0 prediction -> correction -> S1 multiplier -> certificate, against the full-order reference).

Inner product (plan Sec. 3, item 2): ``X = K(mu_bar)`` with ``mu_bar`` the centre of the training box and
``alpha_X(mu) = min_q theta_q(mu) / theta_q(mu_bar)``; for a constant ``K`` this is ``X = K``, ``alpha_X = 1``.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import factorized

from .active_set import by_gap, by_multiplier, recall_and_oversnap
from .bases import dual_basis, pod_basis
from .certificate import certify, is_accepted, min_theta_alpha
from .corrections import RightInverse, repair_r0, repair_r0_snap
from .fom import FomSolution, FullOrderSolver, energy_norm
from .multipliers import MinResidualS1
from .problems import ContactProblem
from .rom import GalerkinROM, ReducedSolution

LOG = logging.getLogger(__name__)
CONTACT_GAP_RTOL = 1e-10          # g^_i <= CONTACT_GAP_RTOL * char_gap counts as "in contact" (coverage diagnostic)
CORRECTIONS = ("R0", "R0s")
ACTIVE_SET_RULES = ("multiplier", "gap", "both")


@dataclass
class Snapshots:
    mus: np.ndarray               # (n, p)
    U: np.ndarray                 # displacement snapshots (n_dof, n)
    L: np.ndarray                 # multiplier snapshots (n_contact, n)
    active: list                  # strictly active sets per snapshot


def solve_many(problem: ContactProblem, mus: np.ndarray, label: str) -> tuple[list[FomSolution], np.ndarray]:
    """Full-order solutions; returns the converged ones and the mask of converged parameters."""
    solver = FullOrderSolver(problem)
    sols, ok = [], np.zeros(len(mus), bool)
    start = time.perf_counter()
    for i, mu in enumerate(mus):
        sol = solver.solve(mu)
        ok[i] = sol.converged
        if sol.converged:
            sols.append(sol)
    LOG.info("%s: %d/%d full-order solves converged in %.1f s", label, ok.sum(), len(mus), time.perf_counter() - start)
    if not ok.all():
        LOG.warning("%s: %d full-order solves failed and are dropped", label, (~ok).sum())
    return sols, ok


def training_snapshots(problem: ContactProblem, rng: np.random.Generator, n: int) -> Snapshots:
    mus = problem.sample(rng, n)
    sols, ok = solve_many(problem, mus, "training")
    return Snapshots(mus[ok], np.column_stack([s.u for s in sols]), np.column_stack([s.lam for s in sols]),
                     [s.strictly_active for s in sols])


def build_bases(snap: Snapshots, r_max: int, r_dual: int, dual_kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Nested Euclidean POD primal basis (first ``r`` columns give the size-``r`` basis) and the dual basis."""
    V = pod_basis(snap.U, r_max)
    if V.shape[1] < r_max:
        LOG.warning("POD rank %d < requested %d", V.shape[1], r_max)
    return V, dual_basis(dual_kind, snap.L, r_dual)


class EnergyInnerProduct:
    """``X = K(mu_bar)`` (factorised once) and the min-theta bound ``alpha_X(mu)``."""

    def __init__(self, problem: ContactProblem, mu_bar: np.ndarray | None = None):
        self.problem = problem
        self.mu_bar = 0.5 * (problem.lo + problem.hi) if mu_bar is None else np.asarray(mu_bar, float)
        self.theta_bar = problem.theta(self.mu_bar)
        self.solve = factorized(problem.stiffness(self.mu_bar).tocsc())

    def alpha(self, mu: np.ndarray) -> float:
        return min_theta_alpha(self.problem.theta(mu), self.theta_bar)


class EuclideanInnerProduct:
    """Baseline ``X = I`` with ``alpha_X = alpha_lb`` (a lower bound of ``lambda_min(K)``); constant ``K`` only.

    For an affine ``K(mu)`` Weyl's bound ``sum_q theta_q lambda_min(K_q)`` is 0 on the layered block (each layer's
    term is singular on the other layer's unknowns), so the baseline is not defined there.
    """

    def __init__(self, problem: ContactProblem, alpha_lb: float):
        if problem.k_varies:
            raise ValueError("the Euclidean baseline is only defined for a constant K here")
        if alpha_lb <= 0:
            raise ValueError("alpha_lb must be positive")
        self.problem, self.alpha_lb = problem, float(alpha_lb)

    @staticmethod
    def solve(r: np.ndarray) -> np.ndarray:
        return np.array(r, dtype=float, copy=True)

    def alpha(self, mu: np.ndarray) -> float:
        return self.alpha_lb


@dataclass
class Certified:
    error: float                  # ||u - u^||_K(mu)
    delta: float                  # certificate
    rho_bar: float
    complementarity: float
    accepted: bool
    covers_plus: bool             # A_+ contained in the contact set of u^ (Corollary E1 / Prop. E3 hypothesis)
    s1_ok: bool
    seconds_correction: float
    seconds_s1: float


class QueryEvaluator:
    """Evaluates G0 and the corrections R0, R0s with the S1 multiplier on one query."""

    def __init__(self, problem: ContactProblem, inner: "EnergyInnerProduct | EuclideanInnerProduct", tau: float,
                 tau_lambda: float,
                 rule: str = "multiplier", tau_gap: float = 0.0):
        if rule not in ACTIVE_SET_RULES:
            raise ValueError(f"unknown active-set rule {rule!r}; choose from {ACTIVE_SET_RULES}")
        self.problem, self.inner, self.tau, self.tau_lambda = problem, inner, tau, tau_lambda
        self.rule, self.tau_gap = rule, tau_gap
        self.right_inverse = RightInverse.from_constraints(problem.B)
        self.s1 = MinResidualS1(problem.B, inner.solve)
        self.contact_tol = CONTACT_GAP_RTOL * problem.char_gap

    def predicted_active(self, red: ReducedSolution, c: np.ndarray) -> np.ndarray:
        """``A^`` from the lifted prediction: ``multiplier``, ``gap``, or ``both`` (intersection of the two)."""
        if self.rule == "multiplier":
            return by_multiplier(red.lam, self.tau_lambda)
        near = by_gap(self.problem.B @ red.u - c, self.tau_gap, self.problem.char_gap)
        if self.rule == "gap":
            return near
        return np.intersect1d(near, by_multiplier(red.lam, self.tau_lambda))

    def correct(self, name: str, red: ReducedSolution, c: np.ndarray, active_hat: np.ndarray) -> np.ndarray:
        B = self.problem.B
        if name == "R0":
            return repair_r0(red.u, B, c, self.right_inverse)
        if name == "R0s":
            return repair_r0_snap(red.u, B, c, self.right_inverse, active_hat)
        raise ValueError(f"unknown correction {name!r}; choose from {CORRECTIONS}")

    def certify_correction(self, name: str, mu: np.ndarray, K, f: np.ndarray, c: np.ndarray, ref: FomSolution,
                           red: ReducedSolution, active_hat: np.ndarray) -> Certified:
        p = self.problem
        start = time.perf_counter()
        u_hat = self.correct(name, red, c, active_hat)
        t_corr = time.perf_counter() - start
        g_hat = np.maximum(p.B @ u_hat - c, 0.0)
        alpha = self.inner.alpha(mu)
        start = time.perf_counter()
        lam_hat, ok = self.s1.solve(f - K @ u_hat, g_hat, alpha)
        t_s1 = time.perf_counter() - start
        cert = certify(K, f, p.B, c, u_hat, lam_hat, self.inner.solve, alpha)
        in_contact = np.flatnonzero(g_hat <= self.contact_tol)
        covers = bool(np.isin(ref.strictly_active, in_contact).all())
        return Certified(energy_norm(K, ref.u - u_hat), cert.delta, cert.rho_bar, cert.complementarity,
                         is_accepted(cert.delta, energy_norm(K, u_hat), self.tau), covers, ok, t_corr, t_s1)

    def active_set_quality(self, red: ReducedSolution, ref: FomSolution, c: np.ndarray) -> tuple[float, float]:
        contact = np.flatnonzero(ref.gap <= self.contact_tol)
        return recall_and_oversnap(self.predicted_active(red, c), ref.strictly_active, contact)


def rom_for_size(problem: ContactProblem, V: np.ndarray, W: np.ndarray, r: int) -> GalerkinROM:
    return GalerkinROM(problem, V[:, :r], W)
