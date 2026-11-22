"""Multiplier choices ``lam^ >= 0`` for the certificate (plan Sec. 3, item 5; MATH_FORMULATION_v6c.md, Sec. 4).

For a feasible ``u^`` let ``a = f - K u^`` and ``g^ = B u^ - c >= 0``, so ``r(lam) = a + B^T lam``. Any ``lam >= 0`` gives
a valid certificate; the variants differ in sharpness and cost.
  * S0            : ``max(lam~, 0)``.
  * S1            : ``argmin_{lam >= 0} Phi(lam) = ||a + B^T lam||^2_{X^-1} / alpha_X + g^ . lam`` (Zhang 2016,
                    (3.123)-(3.124), cited). Strictly convex; equivalent to the LCP
                    ``w = M lam + q``, ``M = 2 B X^-1 B^T / alpha_X``, ``q = 2 B X^-1 a / alpha_X + g^``.
  * S1-restricted : S1 with ``lam_i = 0`` outside a support set (the LCP on the support rows only).
  * S2            : NNLS on the residual ``||a + B^T lam||_{X^-1}``.
Status: all variants done (S1, S1-restricted week 1; S0, S2 week 5). Support rule for S1-restricted:
``active_set.support_by_gap`` (paper Theorem 3.6: the support must contain ``A_+``, else the certificate has a floor).
"""

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp

from .rom import solve_complementarity

VARIANTS = ("S0", "S1", "S1r", "S2")


class MinResidualS1:
    """S1 / S1-restricted for a fixed ``B`` and inner product ``X``.

    Offline: the representers ``X^-1 B^T`` (``n_c`` solves) and ``S_X = B X^-1 B^T``. Per query: one solve for
    ``X^-1 a`` and a dense LCP of size ``n_c`` (or of the support size). For ``X = K`` the LCP matrix is twice the
    Delassus matrix of the condensed full-order solver (formulation Remark 2), so S1 is not cheaper than that solver
    for constant ``K``; Gate 0 times it separately.
    """

    def __init__(self, B: sp.spmatrix, solve_x: Callable[[np.ndarray], np.ndarray]):
        self.B = sp.csr_matrix(B)
        self.solve_x = solve_x
        bt = self.B.T.toarray()
        self.xinv_bt = np.column_stack([solve_x(bt[:, i]) for i in range(bt.shape[1])])
        S = self.B @ self.xinv_bt
        self.S = 0.5 * (S + S.T)

    def solve(self, a: np.ndarray, g_hat: np.ndarray, alpha_x: float,
              support: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """``(lam^, ok)``; ``support`` = row indices allowed to be positive (``None``: all rows)."""
        if alpha_x <= 0:
            raise ValueError("alpha_x must be positive")
        n_c = self.B.shape[0]
        q_full = 2.0 * (self.B @ self.solve_x(a)) / alpha_x + g_hat
        rows = np.arange(n_c) if support is None else np.asarray(support, dtype=int)
        lam = np.zeros(n_c)
        if rows.size == 0:
            return lam, True
        M = (2.0 / alpha_x) * self.S[np.ix_(rows, rows)]
        z, ok = solve_complementarity(M, q_full[rows])
        lam[rows] = np.maximum(z, 0.0)          # round-off only; the LCP solution is non-negative
        return lam, bool(ok)

    @staticmethod
    def objective(residual_dual_norm_sq: float, g_hat: np.ndarray, lam: np.ndarray, alpha_x: float) -> float:
        """``Phi(lam)`` from ``||r(lam)||^2_{X^-1}`` (for tests and diagnostics)."""
        return residual_dual_norm_sq / alpha_x + float(g_hat @ lam)


def clip_s0(lam_tilde: np.ndarray) -> np.ndarray:
    """S0: the lifted ROM multiplier, clipped at zero (no optimisation, cost O(n_c))."""
    return np.maximum(np.asarray(lam_tilde, float), 0.0)


def nnls_s2(s1: MinResidualS1, a: np.ndarray, alpha_x: float,
            support: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """S2: ``argmin_{lam >= 0} ||a + B^T lam||_{X^-1}`` (residual only). It is the S1 problem without the gap term,
    i.e. ``MinResidualS1.solve`` with ``g^ = 0``; the scaling by ``alpha_X`` does not change the minimiser."""
    return s1.solve(a, np.zeros(s1.B.shape[0]), alpha_x, support)
