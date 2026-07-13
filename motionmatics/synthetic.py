"""Synthetic pose sequences.

Real videos are the point of Motionmatics, but for tests, demos, and CI we need
motions whose joint angles we control exactly. This module builds a humanoid
skeleton from a set of joint angles via simple planar forward kinematics, so a
requested "knee = 90°" really does produce a 90° knee that :mod:`angles` will
measure back as 90°.

:func:`demo_pair` returns a matched (reference, user) pair where the user has
deliberate, known faults (shallow squat, under-raised arms, slower tempo) so
the feedback engine has correct answers to find.
"""

from __future__ import annotations

import numpy as np

from . import landmarks as L
from .pose import PoseSequence

# segment lengths (metres, roughly adult proportions)
_TORSO = 0.50
_SHOULDER_HW = 0.18
_HIP_HW = 0.10
_UPPER_ARM = 0.28
_FOREARM = 0.25
_THIGH = 0.42
_SHIN = 0.42
_NECK = 0.12
_HEAD = 0.12


def _rot(vec: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 2-vector (x, y) by ``deg`` degrees (CCW)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def skeleton_from_angles(
    l_shoulder: float, r_shoulder: float,
    l_elbow: float, r_elbow: float,
    l_hip: float, r_hip: float,
    l_knee: float, r_knee: float,
) -> np.ndarray:
    """Build a ``(33, 3)`` world skeleton realising the given joint angles.

    Angle conventions match :mod:`motionmatics.angles`:
    shoulder = arm-to-torso angle, elbow/knee = 3-point angle (180 = straight),
    hip = torso-to-thigh angle (180 = standing tall).
    """
    P = np.zeros((L.NUM_LANDMARKS, 3))

    mid_hip = np.array([0.0, 0.0])
    mid_sh = np.array([0.0, _TORSO])
    l_hip_p = mid_hip + [_HIP_HW, 0.0]
    r_hip_p = mid_hip + [-_HIP_HW, 0.0]
    l_sh_p = mid_sh + [_SHOULDER_HW, 0.0]
    r_sh_p = mid_sh + [-_SHOULDER_HW, 0.0]

    def arm(shoulder_p, hip_p, sh_ang, el_ang, outward_sign):
        down = hip_p - shoulder_p
        down = down / np.linalg.norm(down)
        upper_dir = _rot(down, outward_sign * sh_ang)          # arm-torso angle == sh_ang
        elbow = shoulder_p + _UPPER_ARM * upper_dir
        fore_dir = _rot(upper_dir, outward_sign * (180.0 - el_ang))  # elbow angle == el_ang
        wrist = elbow + _FOREARM * fore_dir
        return elbow, wrist

    def leg(hip_p, shoulder_p, hp_ang, kn_ang, forward_sign):
        # Measure the thigh against the real torso vector (shoulder->hip), which
        # the angle code also uses, so a requested hip angle round-trips exactly.
        up = shoulder_p - hip_p
        up = up / np.linalg.norm(up)
        thigh_dir = _rot(up, 180.0 + forward_sign * (180.0 - hp_ang))  # hip angle == hp_ang
        knee = hip_p + _THIGH * thigh_dir
        shin_dir = _rot(thigh_dir, forward_sign * (180.0 - kn_ang))    # knee angle == kn_ang
        ankle = knee + _SHIN * shin_dir
        return knee, ankle

    l_elbow_p, l_wrist_p = arm(l_sh_p, l_hip_p, l_shoulder, l_elbow, +1)
    r_elbow_p, r_wrist_p = arm(r_sh_p, r_hip_p, r_shoulder, r_elbow, -1)
    l_knee_p, l_ankle_p = leg(l_hip_p, l_sh_p, l_hip, l_knee, +1)
    r_knee_p, r_ankle_p = leg(r_hip_p, r_sh_p, r_hip, r_knee, -1)

    def put(idx, xy, z=0.0):
        P[idx] = [xy[0], xy[1], z]

    put(L.LEFT_HIP, l_hip_p); put(L.RIGHT_HIP, r_hip_p)
    put(L.LEFT_SHOULDER, l_sh_p); put(L.RIGHT_SHOULDER, r_sh_p)
    put(L.LEFT_ELBOW, l_elbow_p); put(L.RIGHT_ELBOW, r_elbow_p)
    put(L.LEFT_WRIST, l_wrist_p); put(L.RIGHT_WRIST, r_wrist_p)
    put(L.LEFT_KNEE, l_knee_p); put(L.RIGHT_KNEE, r_knee_p)
    put(L.LEFT_ANKLE, l_ankle_p); put(L.RIGHT_ANKLE, r_ankle_p)

    # head + face (not used for angles, filled for completeness)
    nose = mid_sh + [0.0, _NECK + _HEAD]
    put(L.NOSE, nose)
    for idx in (L.LEFT_EYE_INNER, L.LEFT_EYE, L.LEFT_EYE_OUTER):
        put(idx, nose + [0.03, 0.02])
    for idx in (L.RIGHT_EYE_INNER, L.RIGHT_EYE, L.RIGHT_EYE_OUTER):
        put(idx, nose + [-0.03, 0.02])
    put(L.LEFT_EAR, nose + [0.06, 0.0]); put(L.RIGHT_EAR, nose + [-0.06, 0.0])
    put(L.MOUTH_LEFT, nose + [0.03, -0.04]); put(L.MOUTH_RIGHT, nose + [-0.03, -0.04])

    # hands + feet, hung off wrist/ankle
    for idx in (L.LEFT_PINKY, L.LEFT_INDEX, L.LEFT_THUMB):
        put(idx, l_wrist_p + [0.0, -0.05])
    for idx in (L.RIGHT_PINKY, L.RIGHT_INDEX, L.RIGHT_THUMB):
        put(idx, r_wrist_p + [0.0, -0.05])
    put(L.LEFT_HEEL, l_ankle_p + [-0.03, -0.03]); put(L.LEFT_FOOT_INDEX, l_ankle_p + [0.08, -0.03])
    put(L.RIGHT_HEEL, r_ankle_p + [0.03, -0.03]); put(L.RIGHT_FOOT_INDEX, r_ankle_p + [-0.08, -0.03])

    # MediaPipe world space has +y downward; flip so "up" reads naturally. Angles
    # are unaffected by a reflection, so this is purely cosmetic for overlays.
    P[:, 1] *= -1.0
    return P


def _bell(t: np.ndarray) -> np.ndarray:
    """A smooth 0→1→0 profile over t in [0,1] (one rep of a movement)."""
    return 0.5 * (1 - np.cos(2 * np.pi * t))


def synth_sequence(
    n_frames: int,
    fps: float,
    *,
    knee_depth: float,
    hip_hinge: float,
    arm_raise: float,
    elbow_bend: float,
    label: str,
    jitter: float = 0.0,
    seed: int = 0,
) -> PoseSequence:
    """One rep: stand → (squat + raise arms) → stand.

    ``knee_depth`` etc. are the *peak* deviations from a standing pose, so larger
    values mean a deeper squat / higher arms.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_frames)
    phase = _bell(t)

    world = np.zeros((n_frames, L.NUM_LANDMARKS, 3))
    for f in range(n_frames):
        p = phase[f]
        knee = 180 - knee_depth * p        # straight -> bent
        hip = 180 - hip_hinge * p          # upright -> hinged
        shoulder = 15 + arm_raise * p      # arms down -> raised
        elbow = 180 - elbow_bend * p       # straight -> slightly bent
        world[f] = skeleton_from_angles(
            shoulder, shoulder, elbow, elbow, hip, hip, knee, knee
        )
        if jitter:
            world[f] += rng.normal(0, jitter, world[f].shape)

    # Fake an image projection: drop z, map metres to a 1280x720-ish [0,1] frame.
    image = np.zeros_like(world)
    xy = world[:, :, :2].copy()
    xy[:, :, 0] = 0.5 + xy[:, :, 0] / 2.0
    xy[:, :, 1] = 0.9 - xy[:, :, 1] / 2.0
    image[:, :, :2] = xy
    visibility = np.ones((n_frames, L.NUM_LANDMARKS))
    return PoseSequence(world=world, image=image, visibility=visibility, fps=fps, label=label)


def demo_pair() -> tuple[PoseSequence, PoseSequence]:
    """A reference rep and a user rep with known, deliberate faults.

    The user: squats shallow (knees under-bent), under-raises the arms, and is
    ~40% slower. The feedback engine should surface exactly these.
    """
    ref = synth_sequence(
        60, 30.0,
        knee_depth=90, hip_hinge=70, arm_raise=150, elbow_bend=20,
        label="reference", jitter=0.002, seed=1,
    )
    user = synth_sequence(
        84, 30.0,  # more frames at same fps => slower
        knee_depth=55, hip_hinge=60, arm_raise=110, elbow_bend=25,
        label="you", jitter=0.004, seed=2,
    )
    return ref, user
