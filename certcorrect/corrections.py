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
Status: R0 and R0s done (week 1); B and Bs in weeks 5-6.
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


def project_energy(u_tilde: np.ndarray, K: sp.spmatrix, B: sp.spmatrix, c: np.ndarray, Z: np.ndarray,
                   active_hat: np.ndarray | None = None) -> np.ndarray:
    """B (``active_hat is None``) or Bs: ``K``-projection onto the feasible set restricted to ``range Z``."""
    raise NotImplementedError("weeks 5-6")
