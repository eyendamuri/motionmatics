"""Tests for the on-device AI coach layer (mocked generation — no model download)."""

import pytest

from motionmatics.ai_coach import AICoachError, ai_advice
from motionmatics.compare import compare_sequences
from motionmatics.synthetic import demo_pair


@pytest.fixture(scope="module")
def report():
    ref, user = demo_pair()
    return compare_sequences(user, ref).report


def test_ai_advice_returns_model_text_and_sends_measurements(report):
    advice_text = "Bend your knees more at the bottom — that's the big one."
    captured = {}

    def fake_generate(messages, max_tokens):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return advice_text

    out = ai_advice(report, activity="squat", _generate=fake_generate)

    assert out == advice_text
    system, user = captured["messages"]
    assert system["role"] == "system" and "coach" in system["content"]
    # the pre-verbalized heuristic facts must reach the model
    user_text = user["content"]
    assert "squat" in user_text
    assert f"{report.score:.0f}/100" in user_text
    assert report.corrections, "demo pair should produce actionable corrections"
    top = report.corrections[0]
    assert top.cue in user_text                                # ranked cue text
    assert f"about {abs(top.mean_signed):.0f} degrees" in user_text  # its magnitude
    assert f"{report.tempo.overall_ratio:.2f} times" in user_text    # tempo fact


def test_ai_advice_raises_on_empty_text(report):
    with pytest.raises(AICoachError):
        ai_advice(report, _generate=lambda messages, max_tokens: "   ")


def test_missing_mlx_gives_helpful_error(report, monkeypatch):
    import sys

    from motionmatics import ai_coach

    ai_coach._load.cache_clear()
    monkeypatch.setitem(sys.modules, "mlx_lm", None)  # forces ImportError
    with pytest.raises(AICoachError, match="mlx-lm"):
        ai_advice(report)
    ai_coach._load.cache_clear()
