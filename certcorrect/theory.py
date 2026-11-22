"""Analysis quantities of ANALYSIS_T1_T3.md, evaluated with the full-order solution (offline checks, not online).

  * T1: missed rows ``M``, contact intensity ``m(M) = ||B^T lam_M||_{K^-1}``, excess identity and bound.
  * T2: snap difference ``p`` (``u^_snap = u^_R0 - B^+ p``) and the constant ``beta = ||K^{1/2} B^+||_2``.
  * T3: floor constant ``sigma_S`` of the restricted multiplier.
"""

from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from .corrections import RightInverse

LANCZOS_TOL = 1e-8


def missed_rows(lam: np.ndarray, g_tilde: np.ndarray, active_hat: np.ndarray) -> np.ndarray:
    """``M = {i in A_+ \\ A^ : g~_i > 0}`` (T1)."""
    candidates = np.flatnonzero((lam > 0) & (g_tilde > 0))
    return np.setdiff1d(candidates, np.asarray(active_hat, dtype=int))


def restrict(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``values`` with the entries outside ``rows`` set to zero."""
    out = np.zeros_like(values)
    out[rows] = values[rows]
    return out


def contact_intensity(B: sp.spmatrix, lam: np.ndarray, rows: np.ndarray,
                      solve_k: Callable[[np.ndarray], np.ndarray]) -> float:
    """``m(rows) = ||B^T lam_rows||_{K^-1}``."""
    v = B.T @ restrict(lam, np.asarray(rows, dtype=int))
    return float(np.sqrt(max(float(v @ solve_k(v)), 0.0)))


def snap_difference(g_tilde: np.ndarray, active_hat: np.ndarray) -> np.ndarray:
    """``p``: ``g~^+`` on ``A^``, zero elsewhere (T2)."""
    return restrict(np.maximum(g_tilde, 0.0), np.asarray(active_hat, dtype=int))


def right_inverse_energy_norm(K: sp.spmatrix, right_inverse: RightInverse) -> float:
    """``beta = ||K^{1/2} B^+||_2 = lambda_max((B^+)^T K B^+)^{1/2}`` (Lanczos)."""
    n_c = right_inverse.B.shape[0]

    def matvec(d):
        w = right_inverse.apply(np.ravel(d))
        return right_inverse.apply_transpose(K @ w)

    op = LinearOperator((n_c, n_c), matvec=matvec, dtype=float)
    top = eigsh(op, k=1, which="LA", tol=LANCZOS_TOL, return_eigenvectors=False)
    return float(np.sqrt(max(top[0], 0.0)))


def support_floor(B: sp.spmatrix, lam: np.ndarray, support: np.ndarray,
                  solve_x: Callable[[np.ndarray], np.ndarray]) -> float:
    """``sigma_S = min_v ||B^T lam_{S^c} - B_S^T v||_{X^-1}`` (T3), by weighted least squares."""
    B = sp.csr_matrix(B)
    support = np.asarray(support, dtype=int)
    outside = np.setdiff1d(np.arange(B.shape[0]), support)
    b = B.T @ restrict(lam, outside)
    xb = solve_x(b)
    if support.size == 0:
        return float(np.sqrt(max(float(b @ xb), 0.0)))
    BS_T = B[support].T.toarray()
    XBS = np.column_stack([solve_x(BS_T[:, j]) for j in range(BS_T.shape[1])])
    v = np.linalg.solve(BS_T.T @ XBS, XBS.T @ b)          # normal equations of the X^-1-weighted problem
    r = b - BS_T @ v
    return float(np.sqrt(max(float(r @ solve_x(r)), 0.0)))
