"""Temporal alignment via Dynamic Time Warping (DTW).

Two people rarely perform a movement at exactly the same speed, and rarely start
on the same frame. DTW finds the monotonic frame-to-frame correspondence between
the user's motion and the reference that minimises the total pose difference,
absorbing differences in tempo. Everything downstream (per-joint error, phase
analysis, tempo) is computed along this warping path.

The implementation is self-contained NumPy — no external DTW dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Alignment:
    """Result of aligning a user sequence to a reference sequence.

    ``path`` is a list of ``(user_idx, ref_idx)`` pairs, monotonically
    non-decreasing in both indices, spanning both sequences end to end.
    """

    path: list[tuple[int, int]]
    cost: float  # total accumulated distance along the path
    normalized_cost: float  # cost divided by path length (avg per-step distance)
    n_user: int
    n_ref: int

    @property
    def user_idx(self) -> np.ndarray:
        return np.array([i for i, _ in self.path], dtype=int)

    @property
    def ref_idx(self) -> np.ndarray:
        return np.array([j for _, j in self.path], dtype=int)


def pairwise_cost(
    user_feat: np.ndarray,
    ref_feat: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted Euclidean distance between every user frame and every ref frame.

    Parameters
    ----------
    user_feat, ref_feat:
        ``(Tu, K)`` and ``(Tr, K)`` feature matrices (e.g. joint angles).
    weights:
        Optional ``(K,)`` per-feature weight (defaults to all ones).

    Returns
    -------
    ``(Tu, Tr)`` distance matrix.
    """
    if weights is None:
        weights = np.ones(user_feat.shape[1])
    w = np.sqrt(np.asarray(weights, dtype=float))
    u = user_feat * w
    r = ref_feat * w
    # ||u_i - r_j||^2 = |u_i|^2 + |r_j|^2 - 2 u_i . r_j
    uu = np.sum(u * u, axis=1)[:, None]
    rr = np.sum(r * r, axis=1)[None, :]
    cross = u @ r.T
    d2 = np.maximum(uu + rr - 2 * cross, 0.0)
    return np.sqrt(d2)


def dtw(cost: np.ndarray, band: float | None = 0.2) -> Alignment:
    """Run DTW over a precomputed ``(Tu, Tr)`` cost matrix.

    Parameters
    ----------
    band:
        Sakoe-Chiba band as a fraction of the longer sequence. Cells whose
        (proportional) position strays further than this from the diagonal are
        forbidden, which both speeds things up and prevents pathological
        warps. ``None`` disables the constraint.
    """
    Tu, Tr = cost.shape
    inf = np.inf
    D = np.full((Tu + 1, Tr + 1), inf)
    D[0, 0] = 0.0

    if band is not None:
        radius = max(int(band * max(Tu, Tr)), abs(Tu - Tr)) + 1
    else:
        radius = max(Tu, Tr)

    for i in range(1, Tu + 1):
        # proportional centre of the band for this row
        center = (i - 1) / max(Tu - 1, 1) * max(Tr - 1, 1)
        j_lo = max(1, int(center) - radius + 1)
        j_hi = min(Tr, int(center) + radius + 1)
        for j in range(j_lo, j_hi + 1):
            c = cost[i - 1, j - 1]
            best = min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            D[i, j] = c + best

    # Backtrace from (Tu, Tr).
    i, j = Tu, Tr
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag, up, left = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        step = min(diag, up, left)
        if step == diag:
            i, j = i - 1, j - 1
        elif step == up:
            i -= 1
        else:
            j -= 1
    path.reverse()

    total = float(D[Tu, Tr])
    return Alignment(
        path=path,
        cost=total,
        normalized_cost=total / max(len(path), 1),
        n_user=Tu,
        n_ref=Tr,
    )


def align_angles(
    user_angles: np.ndarray,
    ref_angles: np.ndarray,
    weights: np.ndarray | None = None,
    band: float | None = 0.2,
) -> Alignment:
    """Convenience wrapper: cost matrix from angle features, then DTW."""
    cost = pairwise_cost(user_angles, ref_angles, weights)
    return dtw(cost, band=band)
