"""AI coaching: turn the heuristic report into natural words of advice.

The heuristic engine (:mod:`motionmatics.feedback`) produces exact, explainable
measurements — "right knee 24° too straight, worst in the middle phase,
1.4× too slow". This module hands that structured output to Claude and asks it
to speak like a coach: prioritized, encouraging, plain-language advice.

The split is deliberate:
  * the *measurements* always come from the deterministic pipeline (the model
    is instructed to use only the numbers it is given, never invent its own),
  * the *phrasing* comes from the LLM, which is what language models are for.

Auth resolves the standard way (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``,
or an ``ant auth login`` profile). Everything degrades gracefully: without the
``anthropic`` package or credentials, callers get an :class:`AICoachError` with
setup instructions and the heuristic report still stands on its own.
"""

from __future__ import annotations

import json

from .feedback import FeedbackReport

DEFAULT_MODEL = "claude-opus-4-8"

_SYSTEM_PROMPT = """\
You are a supportive, precise movement coach. You receive a JSON report from a
motion-comparison system that measured how a person's movement differs from a
reference performance (joint-angle errors in degrees, movement phases, tempo
ratios, and an overall similarity score).

Write the advice you would say to the person, out loud, after watching them.

Rules:
- Use ONLY the measurements in the report. Never invent angles, joints,
  phases, or timing that are not present in it.
- Translate the numbers into feel: "about 25 degrees" -> "noticeably more",
  a tempo ratio of 1.4 -> "you're taking almost half again as long".
  You may still quote a number when it helps.
- Prioritize: lead with the one change that matters most (the corrections
  list is already ranked), then at most two or three more.
- Mention timing only if the tempo is meaningfully off.
- Be encouraging and specific, never generic. No filler like "keep it up!"
  unless the score genuinely warrants it.
- Keep it under 180 words. Plain prose, optionally with a short cue list.
- Do not mention JSON, reports, systems, or that you are an AI.
"""


class AICoachError(RuntimeError):
    """The AI advice step could not run (missing dependency or credentials)."""


def ai_advice(
    report: FeedbackReport,
    *,
    activity: str | None = None,
    model: str = DEFAULT_MODEL,
    client=None,
    max_tokens: int = 16000,
) -> str:
    """Render ``report`` into natural coaching advice via Claude.

    Parameters
    ----------
    report:
        The heuristic :class:`~motionmatics.feedback.FeedbackReport`.
    activity:
        Optional name of the movement (``"squat"``, ``"tennis serve"``) to give
        the coach context; without it the advice stays movement-agnostic.
    client:
        An ``anthropic.Anthropic``-compatible client. Injectable for testing;
        when ``None`` a default client is constructed from the environment.
    """
    if client is None:
        client = _default_client()

    payload = report.to_dict()
    user_text = (
        (f"The movement being compared: {activity}.\n\n" if activity else "")
        + "Measurement report:\n"
        + json.dumps(payload, indent=2)
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )
    except TypeError as e:
        # the SDK validates credentials at request time, not construction
        raise AICoachError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY or run "
            "'ant auth login', then retry with --ai."
        ) from e
    except Exception as e:
        raise AICoachError(f"Claude API call failed: {e}") from e

    if getattr(response, "stop_reason", None) == "refusal":
        raise AICoachError("The model declined to generate advice for this input.")

    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise AICoachError("The model returned no text.")
    return text


def _default_client():
    try:
        import anthropic
    except ImportError as e:
        raise AICoachError(
            "AI advice needs the 'anthropic' package: pip install anthropic"
        ) from e
    try:
        return anthropic.Anthropic()
    except Exception as e:  # missing/invalid credentials
        raise AICoachError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY or run "
            "'ant auth login', then retry with --ai."
        ) from e
