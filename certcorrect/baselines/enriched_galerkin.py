"""Baseline G: Galerkin contact solve on the enriched space ``range Z`` (formulation Remark 4; Fauque-type hybrid
approach, cf. Fauque, Ramiere, Ryckelynck 2018).

    min_q  1/2 q^T H(mu) q - q^T Z^T f(mu)   s.t.  B Z q >= c(mu),     H(mu) = sum_q theta_q Z^T K_q Z,

with the same ``Z`` as the energy projection Bs (``corrections.EnergyProjection``: X-orthonormal span of the ROM basis
``V`` and the compliance columns ``X^-1 B^T``). It differs from B / Bs only in the linear term (the load instead of the
ROM prediction), so it does not use the ROM solution. Solved through the dual LCP
``w = C H^-1 C^T nu + (C H^-1 Z^T f - c) >= 0``, ``nu >= 0``, complementarity (size n_c); the result is cleaned up by R0
(solver-tolerance violations only) so that the feasible-pair certificate applies.
Remark: for a constant K with X = K, Z contains K^-1 B^T; if moreover K^-1 f(mu) lies in range V, the full-order
solution lies in range Z and G reproduces it.
"""

import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from ..corrections import EnergyProjection, repair_r0
from ..rom import solve_complementarity


@dataclass
class GalerkinResult:
    u: np.ndarray                 # feasible displacement (after the R0 clean-up)
    q: np.ndarray                 # coefficients in Z
    nu: np.ndarray                # reduced contact forces (dual LCP solution)
    ok: bool
    repair_norm: float            # ||u - Z q||_K of the clean-up
    seconds: float


class EnrichedGalerkin:
    def __init__(self, projection: EnergyProjection):
        self.projection = projection
        self.problem = projection.problem

    def solve(self, mu: np.ndarray) -> GalerkinResult:
        start = time.perf_counter()
        p, Z, C = self.problem, self.projection.Z, self.projection.C
        H = sum(t * Hq for t, Hq in zip(p.theta(mu), self.projection.H_terms))
        cf = sla.cho_factor(0.5 * (H + H.T))
        Hinv_Ct = sla.cho_solve(cf, C.T)
        Hinv_b = sla.cho_solve(cf, Z.T @ p.load(mu))
        M = C @ Hinv_Ct
        c = p.offset(mu)
        nu, ok = solve_complementarity(0.5 * (M + M.T), C @ Hinv_b - c)
        q = Hinv_b + Hinv_Ct @ nu
        v = Z @ q
        u = repair_r0(v, p.B, c, self.projection.right_inverse)
        K = p.stiffness(mu)
        repair = u - v
        return GalerkinResult(u, q, nu, bool(ok), float(np.sqrt(max(repair @ (K @ repair), 0.0))),
                              time.perf_counter() - start)
