"""On-device AI coaching: turn the heuristic report into natural words of advice.

The heuristic engine (:mod:`motionmatics.feedback`) produces exact, explainable
measurements — "right knee 24° too straight, worst in the middle phase,
1.4× too slow". This module hands that structured output to a small language
model running **locally on your machine** and asks it to speak like a coach:
prioritized, encouraging, plain-language advice.

The split is deliberate:
  * the *measurements* always come from the deterministic pipeline (the model
    is instructed to use only the numbers it is given, never invent its own),
  * the *phrasing* comes from the LLM, which is what language models are for.

Inference uses MLX (Apple's ML framework for Apple Silicon) via ``mlx-lm``.
The default model is Qwen2.5-3B-Instruct quantized to 4 bits — about 1.8 GB,
downloaded once to the Hugging Face cache on first use, fully offline after
that. No API, no key, no data leaves the machine.
"""

from __future__ import annotations

from functools import lru_cache

from .feedback import FeedbackReport

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

_SYSTEM_PROMPT = """\
You are a supportive, precise movement coach. You receive verified coaching
facts from a motion-analysis session: an overall match score, corrections
ranked by importance, and timing notes. Rewrite them as the advice you would
say to the person, out loud, after watching them move.

Rules:
- Every statement must come from the given facts. Do not add corrections,
  numbers, joints, or timing that are not listed.
- Lead with correction #1 — it matters most.
- Weave the facts into natural, encouraging speech; quote a number only when
  it helps ("almost 30 degrees short"). A short numbered cue list is fine.
- Say each correction exactly once, then the timing note, then one closing
  sentence — and stop. Never restate a cue you already gave.
- Keep it under 120 words.
- Do not mention data, facts, reports, systems, or that you are an AI.
"""


class AICoachError(RuntimeError):
    """The AI advice step could not run (missing dependency or model)."""


def ai_advice(
    report: FeedbackReport,
    *,
    activity: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    _generate=None,
) -> str:
    """Render ``report`` into natural coaching advice with an on-device LLM.

    Parameters
    ----------
    report:
        The heuristic :class:`~motionmatics.feedback.FeedbackReport`.
    activity:
        Optional name of the movement (``"squat"``, ``"tennis serve"``) to give
        the coach context; without it the advice stays movement-agnostic.
    model:
        Hugging Face id of an MLX chat model. Downloaded once, cached locally.
    _generate:
        Test seam: a ``fn(messages, max_tokens) -> str`` replacing real
        inference.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_text(report, activity)},
    ]

    if _generate is not None:
        text = _generate(messages, max_tokens)
    else:
        text = _generate_local(model, messages, max_tokens)

    text = (text or "").strip()
    if not text:
        raise AICoachError("The model returned no text.")
    return text


def _user_text(report: FeedbackReport, activity: str | None) -> str:
    """Pre-verbalize the report into plain-English facts.

    A small on-device model shouldn't have to interpret a JSON schema (it will
    misread an error of 28° as an angle of 28°). The heuristic engine already
    knows what each number *means*, so we hand the model ready-made sentences
    and its only job is to speak them like a coach.
    """
    lines: list[str] = []
    if activity:
        lines.append(f"Movement: {activity}.")
    lines.append(f"Overall match with the reference: {report.score:.0f}/100 ({report.grade()}).")

    if report.corrections:
        lines.append("Corrections, ranked most important first:")
        for n, c in enumerate(report.corrections[:3], 1):
            phase = (
                f", worst during the {report.phases[c.worst_phase].name}"
                if report.phases else ""
            )
            lines.append(f"{n}. {c.cue} — off by about {abs(c.mean_signed):.0f} degrees{phase}.")
    else:
        lines.append("No joint corrections needed — all angles match the reference well.")

    t = report.tempo
    if abs(t.overall_ratio - 1.0) >= 0.12:
        speed = "slower" if t.overall_ratio > 1 else "faster"
        lines.append(
            f"Timing: the whole movement was {t.overall_ratio:.2f} times {speed} "
            f"than the reference."
        )
        if report.phases:
            fix = "speed up" if t.worst_phase_ratio > 1 else "slow down"
            lines.append(
                f"Timing was furthest off during the "
                f"{report.phases[t.worst_phase].name} — they should {fix} there."
            )
    else:
        lines.append("Timing: overall tempo matches the reference well.")

    return "\n".join(lines)


@lru_cache(maxsize=1)
def _load(model_id: str):
    try:
        from mlx_lm import load
    except ImportError as e:
        raise AICoachError(
            "On-device AI advice needs the 'mlx-lm' package (Apple Silicon): "
            "pip install mlx-lm"
        ) from e
    try:
        return load(model_id)
    except Exception as e:
        raise AICoachError(
            f"Could not load model '{model_id}' "
            f"(first use downloads ~1 GB — check your connection): {e}"
        ) from e


def _generate_local(model_id: str, messages: list[dict], max_tokens: int) -> str:
    model, tokenizer = _load(model_id)
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        # Greedy (deterministic) decoding: temperature invites invented
        # details and repetition penalties corrupt the numbers, so grounding
        # comes from the model size + pre-verbalized facts instead.
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    except Exception as e:
        raise AICoachError(f"On-device generation failed: {e}") from e
