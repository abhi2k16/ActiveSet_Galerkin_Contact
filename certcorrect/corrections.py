"""Post-solve corrections of the lifted ROM displacement ``u~`` into a feasible ``u^`` (plan Sec. 3, item 4;
MATH_FORMULATION_v6c.md, Secs. 5-6).

  * R0  : ``u^ = u~ + B^+ d``, ``d = max(B u~, t) - B u~`` (``t = c`` plus an optional margin); only penetrating rows move.
  * R0s : R0-snap; in addition ``d_i = t_i - (B u~)_i`` on the predicted active set ``A^``: those rows are put exactly in
          contact (``d_i`` may be negative, i.e. a row floating above the obstacle is pulled down onto it).
  * B   : ``K``-projection of ``u~`` onto ``{v in range Z : B v >= c}`` (small QP).
  * Bs  : B-snap; B with equality rows on ``A^``.
Proposition E3: with ``A_+ subset A^`` the snapped corrections give a full-rate certificate.

``B^+ = B^T (B B^T)^-1`` (Moore-Penrose right inverse, ``B B^+ = I``). For a node-selection ``B`` (one positive entry per
row, distinct columns) ``B B^T`` is diagonal and ``B^+ d`` is computed exactly without factorisation (Prop. 2(iv));
otherwise from the thin QR ``B^T = Q R``, ``B^+ = Q R^-T``.
Status: R0 and R0s done (week 1); B and Bs (``EnergyProjection``) done (week 5).
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


@dataclass
class RightInverse:
    """Applies ``B^+``. ``scale`` is set for a selection ``B`` (``B^+ = B^T diag(1/b_i^2)``), else ``Q, R``."""

    B: sp.csr_matrix
    scale: np.ndarray | None = None
    Q: np.ndarray | None = None
    R: np.ndarray | None = None

    @classmethod
    def from_constraints(cls, B: sp.spmatrix) -> "RightInverse":
        B = sp.csr_matrix(B)
        if is_selection(B):
            return cls(B, scale=1.0 / B.max(axis=1).toarray().ravel() ** 2)
        Q, R = np.linalg.qr(B.T.toarray(), mode="reduced")
        if np.min(np.abs(np.diag(R))) <= np.finfo(float).eps * np.max(np.abs(np.diag(R))) * max(B.shape):
            raise ValueError("B is (numerically) rank deficient: no right inverse")
        return cls(B, Q=Q, R=R)

    @property
    def condition(self) -> float:
        """``kappa(R)`` of the thin QR (``kappa(B)``); reported per the formulation, Sec. 5."""
        if self.scale is not None:
            return float(np.sqrt(self.scale.max() / self.scale.min()))
        return float(np.linalg.cond(self.R))

    def apply(self, d: np.ndarray) -> np.ndarray:
        if self.scale is not None:
            return self.B.T @ (self.scale * d)
        return self.Q @ sla.solve_triangular(self.R, d, trans="T")

    def apply_transpose(self, w: np.ndarray) -> np.ndarray:
        """``(B^+)^T w``."""
        if self.scale is not None:
            return self.scale * (self.B @ w)
        return sla.solve_triangular(self.R, self.Q.T @ w)


def is_selection(B: sp.csr_matrix) -> bool:
    """One positive nonzero per row, in distinct columns."""
    B = sp.csr_matrix(B)
    B.eliminate_zeros()
    if np.any(np.diff(B.indptr) != 1) or np.any(B.data <= 0):
        return False
    return len(np.unique(B.indices)) == B.shape[0]


def repair_r0(u_tilde: np.ndarray, B: sp.spmatrix, c: np.ndarray, right_inverse: RightInverse,
              margin: float = 0.0) -> np.ndarray:
    """R0: lift only the penetrating rows back to ``t = c + margin``."""
    bu = B @ u_tilde
    target = c + margin
    return u_tilde + right_inverse.apply(np.maximum(bu, target) - bu)


def repair_r0_snap(u_tilde: np.ndarray, B: sp.spmatrix, c: np.ndarray, right_inverse: RightInverse,
                   active_hat: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """R0s: R0, plus exact contact (``(B u^)_i = t_i``) on the rows ``active_hat`` (integer indices)."""
    bu = B @ u_tilde
    target = c + margin
    d = np.maximum(bu, target) - bu
    active_hat = np.asarray(active_hat, dtype=int)
    d[active_hat] = target[active_hat] - bu[active_hat]
    return u_tilde + right_inverse.apply(d)


CORRECTION_BASES = ("compliance", "nodal")
ORTH_RELATIVE_CUTOFF = 1e-10      # eigenvalues of the Gram matrix below this fraction of the largest are dropped


@dataclass
class ProjectionResult:
    u: np.ndarray                 # feasible corrected displacement (after the final R0 clean-up)
    ok: bool                      # dual LCP solved
    projection_norm: float        # ||Z delta||_K of the QP correction (formulation Sec. 6, "final repair")
    repair_norm: float            # ||B^+ d||_K of the clean-up that removes solver-tolerance violations


class EnergyProjection:
    """Candidates B and Bs (formulation Sec. 6): the ``K``-projection of ``u~`` onto ``{v in range Z : B v >= c}``,
    for Bs with ``(B v)_i = c_i`` on ``A^``.

    ``Z`` is the ``X``-orthonormalised span of ``[V, W_c]`` with
      * ``compliance``: ``W_c = X^-1 B^T`` (for ``X = K`` the projection is then the exact ``K``-projection onto the
                        full feasible set, since ``P u~ - u~`` lies in ``range K^-1 B^T``);
      * ``nodal``     : ``W_c = B^+`` (corrections through single contact unknowns).
    Either choice contains ``range B^+`` or makes ``B Z`` of full row rank, so the feasible set is nonempty
    (formulation Prop. 5). With ``v = u~ + Z delta`` (``u~ in range V``) the QP is
    ``min 1/2 delta^T H(mu) delta  s.t.  C delta >= -g~``, ``H = Z^T K(mu) Z = sum_q theta_q Z^T K_q Z``, ``C = B Z``,
    solved through its dual LCP of size ``n_c``; the equality rows of Bs are eliminated by a Schur complement.
    """

    def __init__(self, problem, V: np.ndarray, solve_x, X_matrix: sp.spmatrix, right_inverse: RightInverse,
                 basis: str = "compliance", xinv_bt: np.ndarray | None = None):
        if basis not in CORRECTION_BASES:
            raise ValueError(f"unknown correction basis {basis!r}; choose from {CORRECTION_BASES}")
        self.problem, self.right_inverse, self.basis = problem, right_inverse, basis
        B = sp.csr_matrix(problem.B)
        if basis == "compliance":
            extra = xinv_bt if xinv_bt is not None else np.column_stack(
                [solve_x(col) for col in B.T.toarray().T])
        else:
            extra = np.column_stack([right_inverse.apply(col) for col in np.eye(B.shape[0])])
        self.Z = x_orthonormal(np.hstack([V, extra]), X_matrix)
        self.H_terms = tuple(self.Z.T @ (Kq @ self.Z) for Kq in problem.stiffness_terms)
        self.C = np.asarray(B @ self.Z)

    def project(self, mu: np.ndarray, u_tilde: np.ndarray, c: np.ndarray,
                active_hat: np.ndarray | None = None) -> ProjectionResult:
        """B (``active_hat is None``) or Bs."""
        p = self.problem
        H = sum(t * Hq for t, Hq in zip(p.theta(mu), self.H_terms))
        H = 0.5 * (H + H.T)
        Hinv_Ct = sla.cho_solve(sla.cho_factor(H), self.C.T)
        M = self.C @ Hinv_Ct
        M = 0.5 * (M + M.T)
        q = p.B @ u_tilde - c                          # w = M nu + g~ >= 0 (dual of C delta >= -g~)
        equal = np.zeros(0, dtype=int) if active_hat is None else np.asarray(active_hat, dtype=int)
        nu, ok = solve_mixed_lcp(M, q, equal)
        delta = Hinv_Ct @ nu
        v = u_tilde + self.Z @ delta
        K = p.stiffness(mu)
        projection_norm = float(np.sqrt(max(delta @ (H @ delta), 0.0)))
        u_fin = repair_r0(v, p.B, c, self.right_inverse) if active_hat is None else \
            repair_r0_snap(v, p.B, c, self.right_inverse, equal)
        repair = u_fin - v
        return ProjectionResult(u_fin, ok, projection_norm, float(np.sqrt(max(repair @ (K @ repair), 0.0))))


def x_orthonormal(Z0: np.ndarray, X_matrix: sp.spmatrix) -> np.ndarray:
    """Basis of ``range Z0`` with ``Z^T X Z = I`` (method of snapshots; drops numerically dependent directions)."""
    G = Z0.T @ (X_matrix @ Z0)
    w, Q = np.linalg.eigh(0.5 * (G + G.T))
    keep = w > ORTH_RELATIVE_CUTOFF * w.max()
    return Z0 @ (Q[:, keep] / np.sqrt(w[keep]))


def solve_mixed_lcp(M: np.ndarray, q: np.ndarray, equal: np.ndarray) -> tuple[np.ndarray, bool]:
    """``w = M nu + q`` with ``w_E = 0``, ``nu_E`` free on the rows ``equal`` and the usual complementarity elsewhere
    (``M`` symmetric positive definite). The free rows are eliminated by a Schur complement."""
    from .rom import solve_complementarity

    n = len(q)
    if equal.size == 0:
        return solve_complementarity(M, q)
    rest = np.setdiff1d(np.arange(n), equal)
    cf = sla.cho_factor(M[np.ix_(equal, equal)])
    MEE_inv_MER = sla.cho_solve(cf, M[np.ix_(equal, rest)])
    MEE_inv_qE = sla.cho_solve(cf, q[equal])
    nu = np.zeros(n)
    ok = True
    if rest.size:
        S = M[np.ix_(rest, rest)] - M[np.ix_(rest, equal)] @ MEE_inv_MER
        nu_rest, ok = solve_complementarity(0.5 * (S + S.T), q[rest] - M[np.ix_(rest, equal)] @ MEE_inv_qE)
        nu[rest] = nu_rest
    nu[equal] = -MEE_inv_qE - MEE_inv_MER @ nu[rest]
    return nu, bool(ok)
