"""Tests for the AI coach layer (mocked client — no network, no credentials)."""

import json

import pytest

from motionmatics.ai_coach import AICoachError, ai_advice
from motionmatics.compare import compare_sequences
from motionmatics.synthetic import demo_pair


class _Block:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


@pytest.fixture(scope="module")
def report():
    ref, user = demo_pair()
    return compare_sequences(user, ref).report


def test_ai_advice_returns_model_text_and_sends_report(report):
    advice_text = "Bend your knees more at the bottom — that's the big one."
    client = _FakeClient(_Response([_Block("thinking"), _Block("text", advice_text)]))

    out = ai_advice(report, activity="squat", client=client)

    assert out == advice_text
    sent = client.messages.last_kwargs
    assert sent["model"] == "claude-opus-4-8"
    assert sent["thinking"] == {"type": "adaptive"}
    # the full heuristic report must reach the model as valid JSON
    user_text = sent["messages"][0]["content"]
    assert "squat" in user_text
    payload = json.loads(user_text[user_text.index("{"):])
    assert payload == report.to_dict()
    assert payload["corrections"], "demo pair should produce actionable corrections"


def test_ai_advice_raises_on_refusal(report):
    client = _FakeClient(_Response([], stop_reason="refusal"))
    with pytest.raises(AICoachError):
        ai_advice(report, client=client)


def test_ai_advice_raises_on_empty_text(report):
    client = _FakeClient(_Response([_Block("thinking")]))
    with pytest.raises(AICoachError):
        ai_advice(report, client=client)
