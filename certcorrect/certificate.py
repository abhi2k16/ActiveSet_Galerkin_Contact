"""Feasible-pair certificate (Zhang-Bader-Veroy Prop. 4.3; Zhang 2016 Prop. 3.3.6, cited) and the acceptance rule.

For a feasible ``u^`` (``B u^ >= c``) and any ``lam^ >= 0``:
    ||u - u^||_K <= Delta = (rho + sqrt(rho^2 + 4 lam^.g^)) / 2,
    rho = ||f + B^T lam^ - K u^||_{X^-1} / sqrt(alpha_X),   g^ = B u^ - c,
where ``alpha_X`` is a lower bound of the coercivity constant of ``K`` relative to ``X``
(MATH_FORMULATION_v6c.md, Sec. 1). Inner-product choices (plan Sec. 3, item 2):
  * constant ``K``: ``X = K``, ``alpha_X = 1``;
  * affine ``K(mu) = sum_q theta_q(mu) K_q`` with ``theta_q >= 0``: ``X = K(mu_bar)``,
    ``alpha_X = min_q theta_q(mu) / theta_q(mu_bar)`` (``min_theta_alpha``);
  * ``X = I`` (M2 baseline) with a verified ``lambda_min`` bound (``verified.py``).
Accept when ``Delta <= tau / (1 + tau) ||u^||_K``; this implies ``||u - u^||_K <= tau ||u||_K``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

# A pair is treated as feasible when the most negative gap is above -FEASIBILITY_RTOL * ||c||_inf (or the absolute
# floor). Exact feasibility in floating point is the job of verified.py (Level 2); this guard only rejects pairs that
# are clearly infeasible, for which the certificate does not hold.
FEASIBILITY_RTOL = 1e-12
FEASIBILITY_ATOL = 1e-14
NEGATIVE_MULTIPLIER_ATOL = 0.0      # lam^ must be >= 0 exactly; corrections return max(., 0)


@dataclass(frozen=True)
class Certificate:
    delta: float                # certified bound on ||u - u^||_K (inf if the pair is not admissible)
    rho_bar: float              # scaled residual dual norm
    complementarity: float      # lam^ . g^ (>= 0 for an admissible pair)
    alpha_x: float              # coercivity lower bound used
    is_admissible: bool         # u^ feasible and lam^ >= 0 (within the tolerances above)
    min_gap: float              # most negative entry of g^ (diagnostic)


def quadratic_root_bound(rho_bar: float, complementarity: float) -> float:
    """``Delta = (rho + sqrt(rho^2 + 4 s)) / 2``; the positive root of ``Delta^2 - rho Delta - s = 0``."""
    if rho_bar < 0 or complementarity < 0:
        raise ValueError(f"need rho_bar >= 0 and complementarity >= 0, got {rho_bar}, {complementarity}")
    return 0.5 * (rho_bar + np.sqrt(rho_bar**2 + 4.0 * complementarity))


def is_accepted(delta: float, u_hat_norm: float, tau: float) -> bool:
    """Relative acceptance ``Delta <= tau / (1 + tau) ||u^||_K`` (guarantees relative error <= tau)."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return bool(np.isfinite(delta) and delta <= tau / (1.0 + tau) * u_hat_norm)


def min_theta_alpha(theta_mu: np.ndarray, theta_bar: np.ndarray) -> float:
    """Min-theta coercivity bound of ``K(mu)`` relative to ``X = K(mu_bar)`` (needs all coefficients positive)."""
    theta_mu, theta_bar = np.asarray(theta_mu, float), np.asarray(theta_bar, float)
    if theta_mu.shape != theta_bar.shape:
        raise ValueError("theta_mu and theta_bar must have the same shape")
    if np.any(theta_mu < 0) or np.any(theta_bar <= 0):
        raise ValueError("min-theta bound needs theta(mu) >= 0 and theta(mu_bar) > 0")
    return float(np.min(theta_mu / theta_bar))


def dual_norm(residual: np.ndarray, solve_x: Callable[[np.ndarray], np.ndarray]) -> float:
    """``||r||_{X^-1} = sqrt(r^T X^-1 r)``; ``solve_x`` applies ``X^-1`` (e.g. a stored Cholesky factor)."""
    value = float(residual @ solve_x(residual))
    if value < 0:                                     # only possible from round-off for a tiny residual
        value = 0.0
    return float(np.sqrt(value))


def certify(K: sp.spmatrix, f: np.ndarray, B: sp.spmatrix, c: np.ndarray, u_hat: np.ndarray, lam_hat: np.ndarray,
            solve_x: Callable[[np.ndarray], np.ndarray], alpha_x: float) -> Certificate:
    """Full-order (offline / reference) evaluation of the certificate for the pair ``(u^, lam^)``.

    The online evaluation through Riesz representers and Gram blocks (``offline_online.py``) must reproduce this value
    to round-off; tests compare the two.
    """
    if alpha_x <= 0:
        raise ValueError(f"alpha_x must be positive, got {alpha_x}")
    gap = B @ u_hat - c
    min_gap = float(gap.min()) if gap.size else 0.0
    tol = max(FEASIBILITY_RTOL * float(np.linalg.norm(c, np.inf)), FEASIBILITY_ATOL)
    is_admissible = min_gap >= -tol and bool(np.all(lam_hat >= -NEGATIVE_MULTIPLIER_ATOL))
    residual = f + B.T @ lam_hat - K @ u_hat
    rho_bar = dual_norm(residual, solve_x) / np.sqrt(alpha_x)
    complementarity = float(lam_hat @ np.maximum(gap, 0.0))
    delta = quadratic_root_bound(rho_bar, complementarity) if is_admissible else np.inf
    return Certificate(float(delta), float(rho_bar), complementarity, float(alpha_x), is_admissible, min_gap)
