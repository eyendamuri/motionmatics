"""Tests for the data-to-text advice engine (pure Python, no models, no deps)."""

import pytest

from motionmatics.advice import coach_advice
from motionmatics.compare import compare_sequences
from motionmatics.feedback import (
    FeedbackReport,
    JointError,
    PhaseFeedback,
    TempoInfo,
)
from motionmatics.synthetic import demo_pair


@pytest.fixture(scope="module")
def report():
    ref, user = demo_pair()
    return compare_sequences(user, ref).report


def _mk_report(corrections, score=80.0, tempo_ratio=1.0, phases=True):
    ph = (
        [PhaseFeedback(i, n, "left_knee", "", 0.0, 1.0)
         for i, n in enumerate(["start", "middle", "end"])]
        if phases else []
    )
    return FeedbackReport(
        score=score,
        joint_errors=list(corrections),
        corrections=list(corrections),
        phases=ph,
        tempo=TempoInfo(tempo_ratio, 2.0, 2.0 / tempo_ratio, 1, tempo_ratio),
        n_phases=len(ph),
    )


def test_demo_advice_is_grounded_and_aggregated(report):
    text = coach_advice(report, activity="squat")

    assert "squat" in text
    assert f"{report.score:.0f} out of 100" in text
    # the demo pair under-raises BOTH arms -> bilateral aggregation
    assert "both arms" in text
    assert "your left arm" not in text and "your right arm" not in text
    # severities are quoted in degrees and traceable to the report
    top = report.corrections[0]
    assert f"{abs(round(top.mean_signed))}" in text
    assert "degrees" in text
    # 1.4x slower -> "about 40% longer"
    assert "40% longer" in text
    # no unfilled template slots
    assert "{" not in text and "}" not in text


def test_same_report_is_deterministic_but_reports_vary(report):
    assert coach_advice(report) == coach_advice(report)
    assert coach_advice(report, variant=0) != coach_advice(report, variant=1)


def test_single_sided_correction_keeps_the_side():
    err = JointError("left_knee", "left", -20.0, 20.0, 0.9, 1, "bend your left knee more")
    text = coach_advice(_mk_report([err]))
    assert "your left" in text
    assert "both" not in text
    assert "about 20 degrees" in text
    assert "most of all in the middle" in text


def test_clean_report_praises_instead_of_correcting():
    text = coach_advice(_mk_report([], score=95.0))
    assert "95 out of 100" in text
    assert "within tolerance" in text
    assert "off by" not in text


def test_fast_tempo_says_rushing():
    err = JointError("right_elbow", "right", 15.0, 15.0, 0.9, 0, "bend your right elbow more")
    text = coach_advice(_mk_report([err], tempo_ratio=0.7))
    assert "rushing" in text
    assert "30% quicker" in text


def test_mixed_directions_do_not_merge():
    # left arm too low, right arm too high -> opposite directions, no "both"
    lo = JointError("left_shoulder", "left", -25.0, 25.0, 0.9, 1, "")
    hi = JointError("right_shoulder", "right", 25.0, 25.0, 0.9, 1, "")
    text = coach_advice(_mk_report([lo, hi]))
    assert "both arms" not in text
    assert "raise your left arm" in text
    assert "bring your right arm down" in text
