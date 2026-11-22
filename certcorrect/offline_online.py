"""Offline-online evaluation of the certificate (MATH_FORMULATION_v6c.md, offline-online section).

Offline: Riesz representers ``X^-1 [K_q V, B^T, f_q, K_q B^+ columns]`` and their Gram blocks.
Online: ``rho^2`` from the Gram blocks and the reduced coefficients only, with cost independent of ``n_dof`` except for
the repaired columns (rows touched by R0 / R0s).
Must agree with ``certificate.certify`` to round-off (test). Status: weeks 7-8.
"""


class GramBlocks:
    """Precomputed inner products of the residual components in the ``X^-1`` norm."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("weeks 7-8")
