"""Joint-angle features.

Angles are the backbone of Motionmatics' comparison: they are invariant to where
the camera is, how big the person is, and where they stand in frame, so two
people doing the "same" move produce similar angle curves even if they look
different on screen.

All angles are computed from the 3-D *world* landmarks and returned in degrees.
"""

from __future__ import annotations

import numpy as np

from .landmarks import JOINTS, JOINT_NAMES
from .pose import PoseSequence


def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (degrees) at vertex ``b`` in the triangle a-b-c, per frame.

    ``a``, ``b``, ``c`` are ``(T, 3)`` arrays. Returns ``(T,)``.
    """
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba, axis=-1)
    nbc = np.linalg.norm(bc, axis=-1)
    denom = nba * nbc
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / denom
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[denom == 0] = np.nan
    return ang


def joint_angles(seq: PoseSequence) -> dict[str, np.ndarray]:
    """Return ``{joint_name: (T,) degrees}`` for every joint in ``landmarks.JOINTS``."""
    w = seq.world
    out: dict[str, np.ndarray] = {}
    for j in JOINTS:
        out[j.name] = _angle_at_vertex(w[:, j.a], w[:, j.vertex], w[:, j.c])
    return out


def angle_matrix(seq: PoseSequence) -> np.ndarray:
    """Stack the joint angles into a ``(T, n_joints)`` feature matrix.

    Column order follows :data:`landmarks.JOINT_NAMES`. NaNs (from missing
    landmarks) are filled by nearest-valid so the matrix is usable for DTW.
    """
    angles = joint_angles(seq)
    cols = [angles[name] for name in JOINT_NAMES]
    mat = np.stack(cols, axis=1)
    return _fill_nan_columns(mat)


def joint_visibility(seq: PoseSequence) -> np.ndarray:
    """Per-frame, per-joint confidence in ``[0, 1]``.

    A joint's confidence is the minimum visibility of its three landmarks; an
    angle is only trustworthy if all three points are seen.
    """
    vis = seq.visibility
    cols = []
    for j in JOINTS:
        cols.append(np.minimum.reduce([vis[:, j.a], vis[:, j.vertex], vis[:, j.c]]))
    return np.stack(cols, axis=1)


def _fill_nan_columns(mat: np.ndarray) -> np.ndarray:
    mat = mat.copy()
    T, K = mat.shape
    idx = np.arange(T)
    for k in range(K):
        col = mat[:, k]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0.0
        elif mask.any():
            col[mask] = np.interp(idx[mask], idx[~mask], col[~mask])
    return mat
