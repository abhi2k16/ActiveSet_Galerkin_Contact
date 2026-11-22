"""Reduced bases from snapshots.

Primal: proper orthogonal decomposition (truncated SVD) of the displacement snapshots, optionally orthonormal in the
energy inner product of a reference stiffness.
Dual (contact forces), from ``rom_engine``, using only snapshots with contact (``active_snapshot_columns``):
  * ``nmf`` : non-negative matrix factorisation (``nonneg_dual_basis``, NNDSVDa start); ``W lam_r >= 0`` for every
              ``lam_r >= 0``.
  * ``svd`` : leading singular vectors (``svd_dual_basis``); signed, so a lifted multiplier can be negative.
"""

import numpy as np
import scipy.sparse as sp

from rom_engine import active_snapshot_columns, nonneg_dual_basis, svd_dual_basis

DUAL_KINDS = ("nmf", "svd")
SVD_RELATIVE_CUTOFF = 1e-12       # singular values below this fraction of the largest are treated as zero


def pod_basis(snapshots: np.ndarray, r: int, K: sp.spmatrix | None = None) -> np.ndarray:
    """At most ``r`` POD modes of the snapshot columns; Euclidean-orthonormal, or ``K``-orthonormal if ``K`` is given."""
    if r < 1:
        raise ValueError("r must be >= 1")
    S = np.asarray(snapshots, float)
    if K is None:
        modes, sv, _ = np.linalg.svd(S, full_matrices=False)
        return modes[:, :min(r, int(np.sum(sv > SVD_RELATIVE_CUTOFF * sv[0])))]
    # Energy POD via the snapshot Gram matrix (method of snapshots): G = S^T K S = Q diag(s^2) Q^T.
    G = S.T @ (K @ S)
    s2, Q = np.linalg.eigh(0.5 * (G + G.T))
    order = np.argsort(s2)[::-1]
    s2, Q = s2[order], Q[:, order]
    keep = min(r, int(np.sum(s2 > SVD_RELATIVE_CUTOFF**2 * s2[0])))
    return S @ (Q[:, :keep] / np.sqrt(s2[:keep]))


def dual_basis(kind: str, multiplier_snapshots: np.ndarray, r: int) -> np.ndarray:
    """Dual basis with at most ``r`` columns from the snapshots that have contact."""
    if kind not in DUAL_KINDS:
        raise ValueError(f"unknown dual basis kind {kind!r}; choose from {DUAL_KINDS}")
    cols = active_snapshot_columns(multiplier_snapshots)
    if len(cols) == 0:
        raise ValueError("no snapshot with contact: cannot build a dual basis")
    M = np.asarray(multiplier_snapshots, float)[:, cols]
    return nonneg_dual_basis(M, r).U if kind == "nmf" else svd_dual_basis(M, r)
