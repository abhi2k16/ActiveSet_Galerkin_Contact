"""Galerkin contact ROM G0 (Balajewicz et al. 2016) built on ``rom_engine.solve_reduced_contact``.

``u = V u_r``, ``lam = W lam_r``; the reduced problem is
    A_r(mu) u_r = V^T f(mu) + B_r^T lam_r,   B_r u_r >= W^T c(mu),   lam_r >= 0,   complementarity,
with ``A_r(mu) = sum_q theta_q(mu) V^T K_q V`` (terms projected once, offline) and ``B_r = W^T B V``.
For a constant ``K`` this is the same reduced system as ``rom_engine.ContactROM``; the affine split avoids forming
``V^T K(mu) V`` from the full matrix at every query.

The lifted pair ``(u~, lam~)`` may satisfy the reduced conditions and still violate the full ones (``B u~ < c``, or
``lam~ < 0`` for a signed dual basis): the input of the corrections in ``corrections.py``.
"""

import time
from dataclasses import dataclass

import numpy as np

from rom_engine import solve_lcp, solve_reduced_contact

from .problems import ContactProblem


def solve_complementarity(M: np.ndarray, q: np.ndarray, active0: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Dense LCP ``w = M z + q``, ``z, w >= 0``, ``z . w = 0`` for symmetric positive definite ``M``
    (``rom_engine.solve_lcp``, active set with Lemke fallback). Used by the S1 multiplier. Returns ``(z, ok)``."""
    return solve_lcp(M, q, method="active_set", active0=active0)


@dataclass
class ReducedSolution:
    u: np.ndarray                 # lifted displacement u~ = V u_r
    lam: np.ndarray               # lifted multiplier lam~ = W lam_r (may be negative for a signed dual basis)
    u_r: np.ndarray
    lam_r: np.ndarray
    converged: bool               # reduced LCP solved (check before labelling a query)
    seconds: float                # online time: reduced assembly, projection, reduced LCP and lifting


class GalerkinROM:
    """Reduced model for one primal basis ``V`` (n_dof x r) and one dual basis ``W`` (n_contact x r_lam)."""

    def __init__(self, problem: ContactProblem, V: np.ndarray, W: np.ndarray, method: str = "active_set"):
        V, W = np.asarray(V, float), np.asarray(W, float)
        if V.shape[0] != problem.n_dof or W.shape[0] != problem.n_contact:
            raise ValueError(f"basis shapes {V.shape}, {W.shape} do not fit n_dof={problem.n_dof}, "
                             f"n_contact={problem.n_contact}")
        self.problem, self.V, self.W, self.method = problem, V, W, method
        self.A_terms = tuple(V.T @ (Kq @ V) for Kq in problem.stiffness_terms)
        self.B_r = W.T @ (problem.B @ V)

    @property
    def sizes(self) -> tuple[int, int]:
        return self.V.shape[1], self.W.shape[1]

    def reduced_stiffness(self, mu: np.ndarray) -> np.ndarray:
        th = self.problem.theta(mu)
        return sum(coeff * A for coeff, A in zip(th, self.A_terms))

    def solve(self, mu: np.ndarray) -> ReducedSolution:
        p = self.problem
        start = time.perf_counter()
        A_r = self.reduced_stiffness(mu)
        u_r, lam_r, _, ok = solve_reduced_contact(A_r, self.V.T @ p.load(mu), self.B_r, self.W.T @ p.offset(mu),
                                                  self.method)
        u, lam = self.V @ u_r, self.W @ lam_r
        return ReducedSolution(u, lam, u_r, lam_r, bool(ok), time.perf_counter() - start)
