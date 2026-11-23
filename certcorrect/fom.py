"""Full-order contact solve (``fea_engine.solve_contact_static_sparse``, primal-dual active set) and error measures.

Scripts call ``FullOrderSolver.solve``, never the engine directly. Pass ``active0`` (a boolean mask from a previous
solution) to warm-start fallback solves. Check ``FomSolution.converged`` before using a solution as a reference.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from fea_engine import solve_contact_static_sparse

from .problems import ContactProblem

LOG = logging.getLogger(__name__)


@dataclass
class FomSolution:
    u: np.ndarray
    lam: np.ndarray
    gap: np.ndarray                   # B u - c
    active_mask: np.ndarray           # constraints in the solver's final active set
    converged: bool
    iterations: int
    seconds: float
    route: str | None = None
    message: str | None = None
    fallbacks: list = field(default_factory=list)

    @property
    def active(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask)

    @property
    def strictly_active(self) -> np.ndarray:
        """``A_+ = {i : lam_i > 0}`` (the set in Corollary E1 and Proposition E3)."""
        return np.flatnonzero(self.lam > 0)


class FullOrderSolver:
    """Solve ``problem`` at parameters ``mu``."""

    def __init__(self, problem: ContactProblem, route: str = "auto", fallback: bool = True):
        self.problem, self.route, self.fallback = problem, route, fallback

    def solve(self, mu: np.ndarray, active0: np.ndarray | None = None, K: sp.spmatrix | None = None) -> FomSolution:
        """``K`` may be passed when the caller already assembled ``K(mu)``."""
        p = self.problem
        K = p.stiffness(mu) if K is None else K
        f, c = p.load(mu), p.offset(mu)
        start = time.perf_counter()
        res = solve_contact_static_sparse(K, f, p.B, c, route=self.route, active0=active0, fallback=self.fallback)
        seconds = time.perf_counter() - start
        if not res.converged:
            LOG.warning("full-order solve failed at mu=%s: %s", np.array2string(np.asarray(mu), precision=4),
                        res.message)
        mask = np.asarray(res.active_mask, bool) if res.active_mask is not None else res.lam > 0
        return FomSolution(res.u, res.lam, res.gap, mask, bool(res.converged), int(res.iterations), seconds,
                           res.route, res.message, list(res.fallbacks or []))


class CondensedSolver:
    """FOM-cond cost reference for a constant ``K``: ``K`` factorised and the Delassus matrix ``S = B K^-1 B^T`` formed
    once (offline); per query one solve ``K^-1 f``, the dense LCP ``S lam + (B K^-1 f - c) >= 0`` of size ``n_c`` and the
    back-substitution ``u = K^-1 f + (K^-1 B^T) lam``. Exact for the full-order problem (no reduction)."""

    def __init__(self, problem: ContactProblem):
        if problem.k_varies:
            raise ValueError("the condensed solver reuses one factorisation; it needs a constant K")
        from scipy.sparse.linalg import factorized

        from .rom import solve_complementarity

        self.problem, self._lcp = problem, solve_complementarity
        start = time.perf_counter()
        K = problem.stiffness(problem.lo).tocsc()
        self.solve_k = factorized(K)
        bt = problem.B.T.toarray()
        self.KinvBt = np.column_stack([self.solve_k(col) for col in bt.T])
        S = problem.B @ self.KinvBt
        self.S = 0.5 * (S + S.T)
        self.offline_seconds = time.perf_counter() - start

    def solve(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, float]:
        """``(u, lam, converged, seconds)``."""
        p = self.problem
        start = time.perf_counter()
        u0 = self.solve_k(p.load(mu))
        lam, ok = self._lcp(self.S, p.B @ u0 - p.offset(mu))
        u = u0 + self.KinvBt @ lam
        return u, lam, bool(ok), time.perf_counter() - start


def kkt_residuals(K: sp.spmatrix, f: np.ndarray, B: sp.spmatrix, sol: FomSolution) -> dict:
    """The four KKT residuals of a solution (all ~ 0 for a converged full-order solve)."""
    stationarity = K @ sol.u - f - B.T @ sol.lam
    return {"stationarity": float(np.linalg.norm(stationarity, np.inf)),
            "gap_violation": float(max(0.0, -sol.gap.min(initial=0.0))),
            "multiplier_sign": float(max(0.0, -sol.lam.min(initial=0.0))),
            "complementarity": float(np.max(np.abs(sol.lam * sol.gap), initial=0.0))}


def energy_norm(K: sp.spmatrix, v: np.ndarray) -> float:
    return float(np.sqrt(max(float(v @ (K @ v)), 0.0)))


def energy_error(K: sp.spmatrix, u_exact: np.ndarray, u_approx: np.ndarray, relative: bool = True) -> float:
    """``||u - u~||_K``, divided by ``||u||_K`` when ``relative`` (and ``||u||_K > 0``)."""
    err = energy_norm(K, u_exact - u_approx)
    if not relative:
        return err
    denom = energy_norm(K, u_exact)
    return err / denom if denom > 0 else err


def time_call(func, repeats: int = 1) -> tuple[float, object]:
    """``(best wall time in s, last result)`` over ``repeats`` calls."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    best, out = np.inf, None
    for _ in range(repeats):
        start = time.perf_counter()
        out = func()
        best = min(best, time.perf_counter() - start)
    return float(best), out
