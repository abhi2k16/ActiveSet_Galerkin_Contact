"""Summaries computed from saved result arrays (used by experiment logs, figure and table scripts alike)."""

import numpy as np

MIN_POINTS_FOR_SLOPE = 2


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``log y`` against ``log x`` over the finite, positive pairs (nan if fewer than two)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if keep.sum() < MIN_POINTS_FOR_SLOPE or np.ptp(np.log(x[keep])) == 0:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def certificate_rate(error: np.ndarray, delta: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Rate of the certificate in the true error: slope of ``log Delta`` against ``log ||e||`` over all queries and
    basis sizes (1 = full rate, 1/2 = the square-root rate of Proposition E2). ``mask`` selects queries."""
    error, delta = np.asarray(error, float), np.asarray(delta, float)
    if mask is not None:
        error, delta = error[mask], delta[mask]
    return loglog_slope(error.ravel(), delta.ravel())


def best_of(a: dict, methods: tuple[str, ...], suffix: str = "") -> dict[str, np.ndarray]:
    """Per query, the correction with the smallest certificate. Valid: each ``Delta_m`` bounds the error of its own
    ``u^_m``, so returning ``u^_m*`` with ``m* = argmin Delta_m`` is certified by ``min_m Delta_m``.

    Returns ``error``, ``delta``, ``accepted`` and ``chosen`` (index into ``methods``; -1 where no certificate).
    """
    deltas = np.stack([np.where(np.isfinite(a[f"delta_{m}{suffix}"]), a[f"delta_{m}{suffix}"], np.inf)
                       for m in methods])
    errors = np.stack([a[f"error_{m}{suffix}"] for m in methods])
    accepted = np.stack([a[f"accepted_{m}{suffix}"] for m in methods])
    chosen = np.argmin(deltas, axis=0)
    pick = lambda arr: np.take_along_axis(arr, chosen[None], axis=0)[0]          # noqa: E731
    delta = pick(deltas)
    none = ~np.isfinite(delta)
    chosen = np.where(none, -1, chosen)
    error, acc = pick(errors), pick(accepted)
    return {"error": np.where(none, np.nan, error), "delta": np.where(none, np.nan, delta),
            "accepted": np.where(none, 0.0, acc), "chosen": chosen}


def median_by_row(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Median over queries (axis 1) for each basis size, ignoring masked-out and non-finite entries."""
    values = np.asarray(values, float).copy()
    if mask is not None:
        values[~mask] = np.nan
    values[~np.isfinite(values)] = np.nan
    out = np.full(values.shape[0], np.nan)
    for i, row in enumerate(values):
        if np.isfinite(row).any():
            out[i] = np.nanmedian(row)
    return out
