"""Words of advice: a data-to-text engine over the heuristic output.

Turns a :class:`~motionmatics.feedback.FeedbackReport` into the advice a coach
would say out loud. This is deliberately **not** a language model. The input
space is bounded and fully structured (8 joints x 2 directions x severity x
phase x tempo), so the classic NLG pipeline is the better tool:

  content selection -> aggregation -> lexicalization -> realization

* **Grounded by construction**: every clause traces to a report field; there
  is nothing that *can* hallucinate.
* **Generalizes by composition**: any combination of faults yields fluent
  prose because sentences are composed, not retrieved: bilateral faults merge
  ("both arms"), severities pick their own wording, phases attach where they
  belong, and connectives sequence the discourse.
* **Cross-platform and instant**: pure stdlib, no weights, no GPU, no cloud.

Phrasing varies between *different* reports (variant slots picked by a hash of
the report content) but is deterministic for the *same* report, so results are
reproducible and testable.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from .feedback import TEMPO_TOLERANCE, FeedbackReport, JointError

# --- lexicon -----------------------------------------------------------------
# Side-aware and bilateral phrasings per (joint type, direction). "increase"
# means the athlete's angle is smaller than the reference (open it up);
# "decrease" means larger (close it down), mirroring landmarks.Joint.


@dataclass(frozen=True)
class _Lex:
    one: str   # ".. your {side} .." phrasing
    both: str  # bilateral phrasing


_LEXICON: dict[tuple[str, str], _Lex] = {
    ("elbow", "increase"): _Lex("straighten your {side} arm", "straighten both arms"),
    ("elbow", "decrease"): _Lex("bend your {side} elbow more", "bend both elbows more"),
    ("shoulder", "increase"): _Lex(
        "raise your {side} arm higher and further from your body",
        "raise both arms higher and further from your body",
    ),
    ("shoulder", "decrease"): _Lex(
        "bring your {side} arm down closer to your body",
        "bring both arms down closer to your body",
    ),
    ("hip", "increase"): _Lex(
        "open up your {side} hip and stand taller through that side",
        "open up your hips and stand taller",
    ),
    ("hip", "decrease"): _Lex("hinge deeper at your {side} hip", "hinge deeper at your hips"),
    ("knee", "increase"): _Lex("straighten your {side} leg", "straighten both legs"),
    ("knee", "decrease"): _Lex("bend your {side} knee more", "bend both knees more"),
}

# --- variant slots (picked deterministically per report) ----------------------

_OPENERS = {
    "excellent": [
        "That was nearly a mirror image: {score} out of 100.",
        "Beautiful{act}: {score} out of 100, almost identical to the reference.",
    ],
    "good": [
        "Good{act}: {score} out of 100, just a few things to polish.",
        "That's close: {score} out of 100, and the gaps are easy ones.",
    ],
    "fair": [
        "You're getting there{act}: {score} out of 100, so let's tighten a few things.",
        "Decent base{act}: {score} out of 100, with some clear fixes.",
    ],
    "poor": [
        "There's real work to do{act} ({score} out of 100), but the fixes are clear.",
        "Big differences today{act}: {score} out of 100. Here's where to start.",
    ],
}

_FIRST_CONN = ["The biggest thing: ", "Priority one: ", "Start here: "]
_NEXT_CONN = ["Next, ", "After that, ", "Then "]
_LAST_CONN = ["Also, ", "And ", "Lastly, "]

_CLOSERS_WORK = [
    "Work that first cue, then compare again.",
    "Lock in the first fix and re-test; the rest will follow.",
    "Chip away at these in order; the first one moves the score most.",
]
_CLOSERS_CLOSE = [
    "Small stuff; you're close.",
    "Only polish left now.",
]


def _pick(options: list[str], seed: int, slot: int) -> str:
    return options[(seed >> (slot * 4)) % len(options)]


# --- aggregation --------------------------------------------------------------


@dataclass
class _Point:
    """One talking point: a correction, possibly merged across body sides."""

    cue: str
    errors: list[float]      # abs signed error per merged joint, degrees
    phase: str | None        # name of the worst phase (dominant member's)
    rank_key: float          # for ordering (severity x confidence of best member)


def _joint_type(name: str) -> str:
    return name.split("_", 1)[1] if "_" in name else name


def _direction(err: JointError) -> str:
    return "decrease" if err.mean_signed > 0 else "increase"


def _aggregate(report: FeedbackReport, top: int) -> list[_Point]:
    groups: dict[tuple[str, str], list[JointError]] = {}
    order: list[tuple[str, str]] = []
    for c in report.corrections:
        key = (_joint_type(c.name), _direction(c))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(c)

    points: list[_Point] = []
    for key in order:
        members = groups[key]
        best = max(members, key=lambda j: j.mean_abs * j.confidence)
        lex = _LEXICON.get(key)
        if lex is None:  # future joint types: fall back to the engine's cue
            cue = best.cue
        elif len(members) >= 2:
            cue = lex.both
        else:
            cue = lex.one.format(side=best.side)
        phase = report.phases[best.worst_phase].name if report.phases else None
        points.append(
            _Point(
                cue=cue,
                errors=sorted(abs(m.mean_signed) for m in members),
                phase=phase,
                rank_key=best.mean_abs * best.confidence,
            )
        )
    points.sort(key=lambda p: p.rank_key, reverse=True)
    return points[:top]


# --- lexicalization -----------------------------------------------------------


def _degrees_phrase(errors: list[float]) -> str:
    lo, hi = round(errors[0]), round(errors[-1])
    if hi - lo <= 2:
        return f"about {hi} degrees"
    return f"{lo}-{hi} degrees"


def _phase_clause(phase: str | None, seen: set[str]) -> str:
    if phase is None:
        return ""
    if phase in seen:
        return f", again in the {phase}"
    seen.add(phase)
    return f", most of all in the {phase}"


def _score_band(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 55:
        return "fair"
    return "poor"


def _tempo_sentence(report: FeedbackReport) -> str | None:
    t = report.tempo
    if abs(t.overall_ratio - 1.0) < TEMPO_TOLERANCE:
        return "Your timing is right in step with the reference."
    pct = round(abs(t.overall_ratio - 1.0) * 100)
    phase = report.phases[t.worst_phase].name if report.phases else None
    if t.overall_ratio > 1:
        s = f"Timing-wise, you're taking about {pct}% longer than the reference"
        if phase is not None and abs(t.worst_phase_ratio - 1.0) >= TEMPO_TOLERANCE:
            s += f", and the {phase} is where you lose it, so pick up the pace there"
    else:
        s = f"Timing-wise, you're rushing, about {pct}% quicker than the reference"
        if phase is not None and abs(t.worst_phase_ratio - 1.0) >= TEMPO_TOLERANCE:
            s += f", especially the {phase}, so slow that down"
    return s + "."


# --- realization ----------------------------------------------------------------


def coach_advice(
    report: FeedbackReport,
    *,
    activity: str | None = None,
    top: int = 3,
    variant: int | None = None,
) -> str:
    """Compose spoken-style coaching advice from ``report``.

    ``variant`` overrides the deterministic phrasing seed (useful in tests, or
    to re-roll the wording for the same measurements).
    """
    points = _aggregate(report, top)
    seed = variant if variant is not None else _seed(report)

    sentences: list[str] = []

    act = f" on your {activity}" if activity else ""
    opener = _pick(_OPENERS[_score_band(report.score)], seed, 0)
    sentences.append(opener.format(score=f"{report.score:.0f}", act=act))

    if not points:
        sentences.append(
            "Your joint angles all match the reference within tolerance; "
            "this is about polish now, not corrections."
        )
    else:
        seen_phases: set[str] = set()
        for i, p in enumerate(points):
            if i == 0:
                conn = _pick(_FIRST_CONN, seed, 1)
            elif i < len(points) - 1:
                conn = _pick(_NEXT_CONN, seed, 2)
            else:
                conn = _pick(_LAST_CONN, seed, 3)
            sentences.append(
                f"{conn}{p.cue}, off by {_degrees_phrase(p.errors)}"
                f"{_phase_clause(p.phase, seen_phases)}."
            )

    tempo = _tempo_sentence(report)
    if tempo:
        sentences.append(tempo)

    if points:
        closers = _CLOSERS_CLOSE if report.score >= 75 else _CLOSERS_WORK
        sentences.append(_pick(closers, seed, 4))

    return _sentence_case(sentences)


def _seed(report: FeedbackReport) -> int:
    key = "|".join(
        [f"{report.score:.0f}", f"{report.tempo.overall_ratio:.2f}"]
        + [f"{c.name}:{c.mean_signed:.0f}" for c in report.corrections]
    )
    return zlib.crc32(key.encode())


def _sentence_case(sentences: list[str]) -> str:
    out = []
    for s in sentences:
        s = s.strip()
        out.append(s[:1].upper() + s[1:] if s else s)
    return " ".join(out)
