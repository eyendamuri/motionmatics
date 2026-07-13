"""Pose normalization.

MediaPipe *world* landmarks are already hip-centred and metric, but different
people have different limb lengths and may face slightly different directions.
Normalizing to a canonical torso frame makes positional comparisons (used for
overlay drawing and a couple of positional cues) body-size- and
orientation-invariant. Angle features don't need this (angles are already
scale-invariant) but positional features do.
"""

from __future__ import annotations

import numpy as np

from .landmarks import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from .pose import PoseSequence


def torso_scale(world: np.ndarray) -> np.ndarray:
    """Per-frame torso size (mid-shoulder to mid-hip distance), ``(T,)``."""
    mid_sh = 0.5 * (world[:, LEFT_SHOULDER] + world[:, RIGHT_SHOULDER])
    mid_hip = 0.5 * (world[:, LEFT_HIP] + world[:, RIGHT_HIP])
    scale = np.linalg.norm(mid_sh - mid_hip, axis=-1)
    scale[scale < 1e-6] = 1e-6
    return scale


def normalize_world(seq: PoseSequence) -> np.ndarray:
    """Return ``(T, 33, 3)`` world landmarks centred at mid-hip and scaled so the
    torso has unit length. Rotation is left as-is (world landmarks are already
    roughly gravity-aligned).
    """
    world = seq.world
    mid_hip = 0.5 * (world[:, LEFT_HIP] + world[:, RIGHT_HIP])
    centred = world - mid_hip[:, None, :]
    scale = torso_scale(world)[:, None, None]
    return centred / scale
