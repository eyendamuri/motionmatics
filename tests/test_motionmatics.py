"""Tests for the Motionmatics pipeline. Run with ``pytest`` or ``python tests/test_motionmatics.py``."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motionmatics.align import align_angles, dtw, pairwise_cost
from motionmatics.angles import angle_matrix, joint_angles
from motionmatics.compare import compare_sequences
from motionmatics.landmarks import JOINT_NAMES
from motionmatics.pose import PoseSequence
from motionmatics.synthetic import demo_pair, skeleton_from_angles, synth_sequence


def _one_frame(**angles):
    sk = skeleton_from_angles(**angles)
    return PoseSequence(sk[None], sk[None], np.ones((1, 33)), fps=30)


def test_fk_angle_roundtrip():
    """Angles requested in FK are recovered by the angle computation."""
    req = dict(l_shoulder=90, r_shoulder=60, l_elbow=120, r_elbow=150,
               l_hip=150, r_hip=140, l_knee=95, r_knee=100)
    seq = _one_frame(**req)
    a = {k: float(v[0]) for k, v in joint_angles(seq).items()}
    assert abs(a["left_shoulder"] - 90) < 0.5
    assert abs(a["right_elbow"] - 150) < 0.5
    assert abs(a["left_hip"] - 150) < 0.5
    assert abs(a["left_knee"] - 95) < 0.5


def test_angle_matrix_shape_and_order():
    ref, _ = demo_pair()
    mat = angle_matrix(ref)
    assert mat.shape == (ref.num_frames, len(JOINT_NAMES))
    assert not np.isnan(mat).any()


def test_dtw_path_is_monotonic_and_spans():
    a = np.linspace(0, 1, 40)[:, None] * np.array([1.0, 2.0])
    b = np.linspace(0, 1, 25)[:, None] * np.array([1.0, 2.0])
    al = align_angles(a, b, band=0.3)
    ui, ri = al.user_idx, al.ref_idx
    assert ui[0] == 0 and ri[0] == 0
    assert ui[-1] == 39 and ri[-1] == 24
    assert np.all(np.diff(ui) >= 0) and np.all(np.diff(ri) >= 0)


def test_identical_sequences_score_high():
    ref, _ = demo_pair()
    res = compare_sequences(ref, ref)
    assert res.report.score > 97
    assert len(res.report.corrections) == 0
    assert abs(res.report.tempo.overall_ratio - 1.0) < 0.05


def test_demo_pair_finds_known_faults():
    """User squats shallow, under-raises arms, and is slower — all must surface."""
    ref, user = demo_pair()
    res = compare_sequences(user, ref)
    errs = {j.name: j for j in res.report.joint_errors}

    # knees not bent enough => user angle larger => positive signed error
    assert errs["left_knee"].mean_signed > 8
    assert errs["right_knee"].mean_signed > 8
    # arms under-raised => user shoulder angle smaller => negative signed error
    assert errs["left_shoulder"].mean_signed < -8
    # the top cues mention knees and/or arms
    cues = " ".join(c.cue for c in res.report.corrections).lower()
    assert "knee" in cues and "arm" in cues
    # user is slower than reference
    assert res.report.tempo.overall_ratio > 1.2


def test_faster_user_detected():
    ref = synth_sequence(60, 30.0, knee_depth=80, hip_hinge=60, arm_raise=140,
                         elbow_bend=20, label="reference")
    fast = synth_sequence(36, 30.0, knee_depth=80, hip_hinge=60, arm_raise=140,
                          elbow_bend=20, label="you")
    res = compare_sequences(fast, ref)
    assert res.report.tempo.overall_ratio < 0.8  # faster


def test_pairwise_cost_zero_on_self():
    a = np.random.default_rng(0).normal(size=(10, 8))
    c = pairwise_cost(a, a)
    # expanded-norm formula has ~1e-7 float cancellation on the diagonal
    assert np.allclose(np.diag(c), 0, atol=1e-4)


def test_score_monotonic_in_error():
    """A bigger amplitude fault should score lower."""
    ref = synth_sequence(60, 30.0, knee_depth=90, hip_hinge=70, arm_raise=150,
                         elbow_bend=20, label="ref")
    small = synth_sequence(60, 30.0, knee_depth=80, hip_hinge=65, arm_raise=140,
                           elbow_bend=20, label="you")
    big = synth_sequence(60, 30.0, knee_depth=30, hip_hinge=30, arm_raise=70,
                         elbow_bend=20, label="you")
    s_small = compare_sequences(small, ref).report.score
    s_big = compare_sequences(big, ref).report.score
    assert s_small > s_big


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
