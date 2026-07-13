"""The heuristic coaching engine.

Given the user's and reference's joint-angle curves aligned by DTW, this module
decides *what the user should change* to look more like the reference. It is
deliberately rule-based and explainable: every cue traces back to a measured
angle difference in degrees, a body part, and a moment in the movement.

Pipeline:
  1. Walk the DTW path; at every matched pair measure each joint's angle error
     (user minus reference), weighted by how confidently both bodies were seen.
  2. Aggregate per joint: mean signed error (direction) + mean |error| (severity).
  3. Any joint whose error exceeds a tolerance becomes a ranked correction, with
     a plain-language cue ("bend your right knee more") and the degrees involved.
  4. Split the movement into phases and find the dominant fault in each.
  5. Compare tempo (overall and per phase) from the warp path.
  6. Map the residual error to a 0-100 similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import Alignment
from .landmarks import JOINTS_BY_NAME, JOINT_NAMES

# --- tunable heuristics -----------------------------------------------------
ANGLE_TOLERANCE_DEG = 8.0  # differences smaller than this are "close enough"
SCORE_GOOD_ERR = 5.0       # per-joint error (deg) that scores ~100
SCORE_BAD_ERR = 40.0       # per-joint error (deg) that scores ~0
MIN_CONFIDENCE = 0.3       # ignore joints seen less confidently than this
TEMPO_TOLERANCE = 0.12     # |local tempo ratio - 1| below this is "in time"

# Coaching cares most about the moments of *exertion* (the bottom of a squat,
# the top of an arm raise), not the frames where the athlete is standing at
# rest. We therefore weight each frame's error by how far the reference joint is
# from its anatomical neutral (plus a floor so a joint held statically in the
# wrong place still registers). NEUTRAL is the "rest" angle per joint type.
ACTIVITY_FLOOR_DEG = 12.0
_NEUTRAL_BY_TYPE = {"elbow": 180.0, "shoulder": 10.0, "hip": 180.0, "knee": 180.0}


def _neutral_for(joint_name: str) -> float:
    for key, val in _NEUTRAL_BY_TYPE.items():
        if joint_name.endswith(key):
            return val
    return 90.0


@dataclass
class JointError:
    name: str
    side: str
    mean_signed: float   # user - ref, degrees (sign carries direction)
    mean_abs: float      # severity, degrees
    confidence: float    # 0..1, how reliably this joint was tracked
    worst_phase: int     # index of the phase with the largest |error|
    cue: str             # plain-language correction ("" if within tolerance)

    @property
    def actionable(self) -> bool:
        return self.mean_abs >= ANGLE_TOLERANCE_DEG and self.confidence >= MIN_CONFIDENCE


@dataclass
class PhaseFeedback:
    index: int
    name: str
    dominant_joint: str
    dominant_cue: str
    dominant_error: float
    tempo_ratio: float   # user time / ref time within this phase (>1 == slower)


@dataclass
class TempoInfo:
    overall_ratio: float  # user duration / ref duration (>1 == slower overall)
    user_duration: float
    ref_duration: float
    worst_phase: int
    worst_phase_ratio: float


@dataclass
class FeedbackReport:
    score: float
    joint_errors: list[JointError]
    corrections: list[JointError]  # actionable, ranked by severity
    phases: list[PhaseFeedback]
    tempo: TempoInfo
    n_phases: int
    user_label: str = "you"
    ref_label: str = "reference"
    meta: dict = field(default_factory=dict)

    # -- rendering ----------------------------------------------------------
    def grade(self) -> str:
        s = self.score
        if s >= 90:
            return "Excellent, nearly identical"
        if s >= 75:
            return "Good, a few adjustments"
        if s >= 55:
            return "Fair, several things to fix"
        return "Needs work, big differences"

    def render_text(self, top: int = 4) -> str:
        lines: list[str] = []
        lines.append(f"Motion match: {self.score:.0f}/100  ({self.grade()})")
        lines.append("")

        if not self.corrections:
            lines.append("No major corrections: your angles are within tolerance of the")
            lines.append("reference. Focus on matching the tempo below.")
        else:
            lines.append("Top corrections:")
            for n, c in enumerate(self.corrections[:top], 1):
                direction = "too straight/open" if c.mean_signed > 0 else "too bent/closed"
                phase = self.phases[c.worst_phase].name if self.phases else "throughout"
                lines.append(
                    f" {n}. {_cap(c.cue)}, about {abs(c.mean_signed):.0f}° "
                    f"{direction} (worst {phase})."
                )
        lines.append("")

        # tempo
        t = self.tempo
        if abs(t.overall_ratio - 1.0) < TEMPO_TOLERANCE:
            lines.append("Timing: your overall tempo matches the reference well.")
        else:
            faster_slower = "slower" if t.overall_ratio > 1 else "faster"
            lines.append(
                f"Timing: your motion is {t.overall_ratio:.2f}× {faster_slower} "
                f"than the {self.ref_label} overall."
            )
        if self.phases and abs(t.worst_phase_ratio - 1.0) >= TEMPO_TOLERANCE:
            ph = self.phases[t.worst_phase]
            adj = "speed up" if t.worst_phase_ratio > 1 else "slow down"
            lines.append(f"        You're most out of time in the {ph.name}, so try to {adj} there.")
        lines.append("")

        # phase-by-phase
        if self.phases:
            lines.append("Phase-by-phase:")
            for ph in self.phases:
                if ph.dominant_cue:
                    lines.append(
                        f" - {_cap(ph.name)}: {ph.dominant_cue} "
                        f"(≈{ph.dominant_error:.0f}° off)."
                    )
                else:
                    lines.append(f" - {_cap(ph.name)}: looking good.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "grade": self.grade(),
            "user_label": self.user_label,
            "ref_label": self.ref_label,
            "corrections": [
                {
                    "joint": c.name,
                    "cue": c.cue,
                    "mean_signed_deg": round(c.mean_signed, 1),
                    "mean_abs_deg": round(c.mean_abs, 1),
                    "confidence": round(c.confidence, 2),
                    "worst_phase": self.phases[c.worst_phase].name if self.phases else None,
                }
                for c in self.corrections
            ],
            "joint_errors": [
                {
                    "joint": j.name,
                    "mean_signed_deg": round(j.mean_signed, 1),
                    "mean_abs_deg": round(j.mean_abs, 1),
                    "confidence": round(j.confidence, 2),
                    "actionable": j.actionable,
                }
                for j in self.joint_errors
            ],
            "tempo": {
                "overall_ratio": round(self.tempo.overall_ratio, 3),
                "user_duration_s": round(self.tempo.user_duration, 2),
                "ref_duration_s": round(self.tempo.ref_duration, 2),
                "worst_phase": (
                    self.phases[self.tempo.worst_phase].name if self.phases else None
                ),
                "worst_phase_ratio": round(self.tempo.worst_phase_ratio, 3),
            },
            "phases": [
                {
                    "name": p.name,
                    "dominant_joint": p.dominant_joint,
                    "dominant_cue": p.dominant_cue,
                    "dominant_error_deg": round(p.dominant_error, 1),
                    "tempo_ratio": round(p.tempo_ratio, 3),
                }
                for p in self.phases
            ],
            "meta": self.meta,
        }


_PHASE_NAMES_3 = ["start", "middle", "end"]


def _phase_names(n: int) -> list[str]:
    if n == 3:
        return list(_PHASE_NAMES_3)
    if n == 2:
        return ["first half", "second half"]
    return [f"phase {i + 1}" for i in range(n)]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def generate_feedback(
    user_angles: np.ndarray,
    ref_angles: np.ndarray,
    alignment: Alignment,
    user_conf: np.ndarray,
    ref_conf: np.ndarray,
    user_fps: float,
    ref_fps: float,
    n_phases: int = 3,
    user_label: str = "you",
    ref_label: str = "reference",
) -> FeedbackReport:
    """Build a :class:`FeedbackReport` from aligned angle curves.

    Parameters
    ----------
    user_angles, ref_angles:
        ``(Tu, K)`` / ``(Tr, K)`` joint-angle matrices (degrees), columns in
        :data:`landmarks.JOINT_NAMES` order.
    alignment:
        The DTW result linking user frames to reference frames.
    user_conf, ref_conf:
        ``(Tu, K)`` / ``(Tr, K)`` per-frame per-joint confidence in ``[0, 1]``.
    """
    ui = alignment.user_idx
    ri = alignment.ref_idx
    K = user_angles.shape[1]
    P = max(1, n_phases)
    names = _phase_names(P)

    # Assign each aligned pair to a phase by its position on the REFERENCE
    # timeline (the reference defines the canonical structure of the move).
    ref_pos = ri / max(alignment.n_ref - 1, 1)
    phase_of_pair = np.clip((ref_pos * P).astype(int), 0, P - 1)

    # Per pair: signed error and weight for every joint. The weight combines
    # tracking confidence with reference "activity" (distance from the joint's
    # neutral pose), so exertion moments dominate and idle frames don't.
    err = user_angles[ui] - ref_angles[ri]            # (M, K)
    conf = np.minimum(user_conf[ui], ref_conf[ri])    # (M, K)
    neutral = np.array([_neutral_for(n) for n in JOINT_NAMES])  # (K,)
    activity = np.abs(ref_angles[ri] - neutral)       # (M, K)
    weight = conf * (ACTIVITY_FLOOR_DEG + activity)   # (M, K)

    joint_errors: list[JointError] = []
    # accumulate per-phase per-joint stats for phase analysis
    phase_abs = np.zeros((P, K))
    phase_wsum = np.zeros((P, K))

    for p in range(P):
        m = phase_of_pair == p
        if m.any():
            w = weight[m]
            e = np.abs(err[m])
            phase_wsum[p] = w.sum(axis=0)
            phase_abs[p] = (w * e).sum(axis=0)

    for k, name in enumerate(JOINT_NAMES):
        w = weight[:, k]
        wsum = w.sum()
        # Tracking confidence is the raw per-joint visibility, independent of the
        # activity weighting used for the error means.
        confidence = float(conf[:, k].mean())
        if wsum <= 1e-6 or confidence < 1e-6:
            joint_errors.append(JointError(name, JOINTS_BY_NAME[name].side, 0, 0, confidence, 0, ""))
            continue
        mean_signed = float((w * err[:, k]).sum() / wsum)
        mean_abs = float((w * np.abs(err[:, k])).sum() / wsum)

        # worst phase for this joint = phase with largest mean |error|
        with np.errstate(invalid="ignore", divide="ignore"):
            phase_mean = np.where(phase_wsum[:, k] > 1e-6, phase_abs[:, k] / phase_wsum[:, k], 0.0)
        worst_phase = int(np.argmax(phase_mean))

        joint = JOINTS_BY_NAME[name]
        cue = ""
        if mean_abs >= ANGLE_TOLERANCE_DEG and confidence >= MIN_CONFIDENCE:
            cue = joint.decrease if mean_signed > 0 else joint.increase

        joint_errors.append(
            JointError(name, joint.side, mean_signed, mean_abs, confidence, worst_phase, cue)
        )

    corrections = sorted(
        (j for j in joint_errors if j.actionable),
        key=lambda j: j.mean_abs * j.confidence,
        reverse=True,
    )

    tempo = _tempo(alignment, user_fps, ref_fps, phase_of_pair, P)
    phases = _phase_feedback(joint_errors, phase_abs, phase_wsum, names, tempo)
    score = _score(joint_errors)

    return FeedbackReport(
        score=score,
        joint_errors=joint_errors,
        corrections=corrections,
        phases=phases,
        tempo=tempo,
        n_phases=P,
        user_label=user_label,
        ref_label=ref_label,
        meta={"n_matched_pairs": len(alignment.path), "dtw_cost": round(alignment.normalized_cost, 3)},
    )


def _tempo(
    alignment: Alignment,
    user_fps: float,
    ref_fps: float,
    phase_of_pair: np.ndarray,
    P: int,
) -> TempoInfo:
    user_dur = alignment.n_user / (user_fps or 30.0)
    ref_dur = alignment.n_ref / (ref_fps or 30.0)
    overall = user_dur / ref_dur if ref_dur > 0 else 1.0

    ui, ri = alignment.user_idx, alignment.ref_idx
    worst_phase, worst_ratio, worst_dev = 0, 1.0, -1.0
    for p in range(P):
        m = phase_of_pair == p
        if m.sum() < 2:
            continue
        user_span = (ui[m].max() - ui[m].min() + 1) / (user_fps or 30.0)
        ref_span = (ri[m].max() - ri[m].min() + 1) / (ref_fps or 30.0)
        ratio = user_span / ref_span if ref_span > 0 else 1.0
        dev = abs(ratio - 1.0)
        if dev > worst_dev:
            worst_dev, worst_ratio, worst_phase = dev, ratio, p
    return TempoInfo(overall, user_dur, ref_dur, worst_phase, worst_ratio)


def _phase_feedback(
    joint_errors: list[JointError],
    phase_abs: np.ndarray,
    phase_wsum: np.ndarray,
    names: list[str],
    tempo: TempoInfo,
) -> list[PhaseFeedback]:
    phases: list[PhaseFeedback] = []
    P = len(names)
    with np.errstate(invalid="ignore", divide="ignore"):
        phase_mean = np.where(phase_wsum > 1e-6, phase_abs / phase_wsum, 0.0)  # (P, K)
    for p in range(P):
        row = phase_mean[p]
        k = int(np.argmax(row)) if row.size else 0
        dominant_err = float(row[k]) if row.size else 0.0
        name = JOINT_NAMES[k]
        joint = JOINTS_BY_NAME[name]
        cue = ""
        if dominant_err >= ANGLE_TOLERANCE_DEG:
            # direction from that joint's overall signed error
            signed = joint_errors[k].mean_signed
            cue = joint.decrease if signed > 0 else joint.increase
        # per-phase tempo ratio (recomputed same way as in _tempo, kept simple)
        ratio = tempo.worst_phase_ratio if p == tempo.worst_phase else 1.0
        phases.append(PhaseFeedback(p, names[p], name, cue, dominant_err, ratio))
    return phases


def _score(joint_errors: list[JointError]) -> float:
    """Map residual per-joint error to 0-100, averaged over tracked joints."""
    scores, weights = [], []
    for j in joint_errors:
        if j.confidence < MIN_CONFIDENCE:
            continue
        s = 100.0 * (1.0 - (j.mean_abs - SCORE_GOOD_ERR) / (SCORE_BAD_ERR - SCORE_GOOD_ERR))
        scores.append(float(np.clip(s, 0.0, 100.0)))
        weights.append(j.confidence)
    if not scores:
        return 0.0
    return float(np.average(scores, weights=weights))
