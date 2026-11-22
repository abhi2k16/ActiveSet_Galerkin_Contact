"""Finite-precision questions of the certificate.

Now (week 2): ``extreme_eigenvalues`` gives Lanczos (shift-invert) ESTIMATES of ``lambda_min`` and ``lambda_max`` of an
SPD matrix, used only by the Euclidean baseline ``X = I`` in X2. A Ritz value is an upper estimate of ``lambda_min``,
so it is NOT a valid coercivity lower bound (MATH_FORMULATION_v6c.md, Sec. 3, item 4). Using it favours the baseline
(a smaller certificate than a verified bound would give); results with it are labelled "unverified".

Week 13 (X8): Level 2 rounding bounds on ``rho``, ``lam^.g^`` and feasibility, and a verified lower bound of
``lambda_min`` (verified Cholesky of ``K - beta I`` with bisection on ``beta``).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

LANCZOS_TOL = 1e-10


@dataclass(frozen=True)
class EigenEstimate:
    lambda_min: float             # Ritz value (upper estimate of the true lambda_min; unverified)
    lambda_max: float
    residual_min: float           # ||K x - lambda x|| / |lambda| of the lambda_min Ritz pair
    is_verified: bool = False

    @property
    def sqrt_condition(self) -> float:
        """``sqrt(kappa(K))``: the effectivity factor of ``X = I`` (formulation Remark 1)."""
        return float(np.sqrt(self.lambda_max / self.lambda_min))


def extreme_eigenvalues(K: sp.spmatrix) -> EigenEstimate:
    """Lanczos estimates of the extreme eigenvalues of SPD ``K`` (shift-invert at 0 for the smallest)."""
    K = sp.csc_matrix(K)
    lam_min, vec = eigsh(K, k=1, sigma=0.0, which="LM", tol=LANCZOS_TOL)
    lam_max = eigsh(K, k=1, which="LA", tol=LANCZOS_TOL, return_eigenvectors=False)
    x = vec[:, 0]
    if lam_min[0] <= 0:
        raise ValueError(f"matrix is not positive definite (Ritz value {lam_min[0]:.3e})")
    residual = float(np.linalg.norm(K @ x - lam_min[0] * x) / lam_min[0])
    return EigenEstimate(float(lam_min[0]), float(lam_max[0]), residual)


def verified_lambda_min_lower(*args, **kwargs) -> float:
    raise NotImplementedError("week 13")


def verified_certificate(*args, **kwargs):
    raise NotImplementedError("week 13")
