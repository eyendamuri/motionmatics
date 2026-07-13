"""End-to-end comparison: two motions in, a coaching report out.

    reference video ─┐
                     ├─► poses ─► angles ─► DTW align ─► heuristic feedback
    your video ──────┘
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .align import Alignment, align_angles
from .angles import angle_matrix, joint_visibility
from .feedback import FeedbackReport, generate_feedback
from .pose import PoseExtractor, PoseSequence


def smooth_columns(mat: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving-average smoothing of each column to tame per-frame jitter."""
    if window <= 1 or mat.shape[0] < window:
        return mat
    kernel = np.ones(window) / window
    out = np.empty_like(mat)
    pad = window // 2
    for k in range(mat.shape[1]):
        padded = np.pad(mat[:, k], pad, mode="edge")
        out[:, k] = np.convolve(padded, kernel, mode="valid")[: mat.shape[0]]
    return out


def standardize_columns(mat: np.ndarray, std_floor: float = 5.0) -> np.ndarray:
    """Per-column z-score (within one clip) for *alignment* features.

    Aligning on the standardized shape of each movement — rather than the raw
    angles — stops DTW from hiding amplitude faults by matching a shallow squat
    to the reference's mid-depth frames. After alignment we always measure error
    on the raw angles, so the true amplitude gap is preserved. ``std_floor``
    keeps near-static joints from being blown up into noise.
    """
    mean = mat.mean(axis=0, keepdims=True)
    std = np.maximum(mat.std(axis=0, keepdims=True), std_floor)
    return (mat - mean) / std


@dataclass
class ComparisonResult:
    report: FeedbackReport
    alignment: Alignment
    user_angles: np.ndarray
    ref_angles: np.ndarray
    user_seq: PoseSequence
    ref_seq: PoseSequence


def compare_sequences(
    user_seq: PoseSequence,
    ref_seq: PoseSequence,
    n_phases: int = 3,
    smooth: int = 5,
    band: float | None = 0.2,
) -> ComparisonResult:
    """Compare two already-extracted pose sequences."""
    user_ang = smooth_columns(angle_matrix(user_seq), smooth)
    ref_ang = smooth_columns(angle_matrix(ref_seq), smooth)
    user_conf = joint_visibility(user_seq)
    ref_conf = joint_visibility(ref_seq)

    # Weight the alignment toward joints that are reliably visible in BOTH clips.
    weights = np.minimum(user_conf.mean(axis=0), ref_conf.mean(axis=0))
    weights = weights / (weights.sum() or 1.0) * len(weights)

    # Align on standardized shape (timing), then score on raw angles (amplitude).
    alignment = align_angles(
        standardize_columns(user_ang),
        standardize_columns(ref_ang),
        weights=weights,
        band=band,
    )

    report = generate_feedback(
        user_ang, ref_ang, alignment,
        user_conf=user_conf, ref_conf=ref_conf,
        user_fps=user_seq.fps, ref_fps=ref_seq.fps,
        n_phases=n_phases,
        user_label=user_seq.label, ref_label=ref_seq.label,
    )
    return ComparisonResult(report, alignment, user_ang, ref_ang, user_seq, ref_seq)


def compare_videos(
    user_video: str,
    ref_video: str,
    model_path: str | None = None,
    n_phases: int = 3,
    smooth: int = 5,
    band: float | None = 0.2,
    max_frames: int | None = None,
    progress: bool = True,
) -> ComparisonResult:
    """Extract poses from two videos and compare them."""
    extractor = PoseExtractor(model_path=model_path)
    if progress:
        print("Extracting poses from reference…", flush=True)
    ref_seq = extractor.extract(ref_video, label="reference", max_frames=max_frames, progress=progress)
    if progress:
        print("Extracting poses from your video…", flush=True)
    user_seq = extractor.extract(user_video, label="you", max_frames=max_frames, progress=progress)
    return compare_sequences(user_seq, ref_seq, n_phases=n_phases, smooth=smooth, band=band)
