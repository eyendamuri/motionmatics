"""Landmark topology for MediaPipe Pose (33 keypoints) and the joint/segment
definitions Motionmatics uses to reason about a body's configuration.

MediaPipe returns 33 landmarks per frame. We only care about the "big" joints
that carry the shape of a movement (elbows, shoulders, hips, knees) plus a few
segments used for normalization and framing.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The 33 MediaPipe Pose landmarks, by index.
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

NUM_LANDMARKS = 33

LANDMARK_NAMES = {
    NOSE: "nose",
    LEFT_SHOULDER: "left shoulder",
    RIGHT_SHOULDER: "right shoulder",
    LEFT_ELBOW: "left elbow",
    RIGHT_ELBOW: "right elbow",
    LEFT_WRIST: "left wrist",
    RIGHT_WRIST: "right wrist",
    LEFT_HIP: "left hip",
    RIGHT_HIP: "right hip",
    LEFT_KNEE: "left knee",
    RIGHT_KNEE: "right knee",
    LEFT_ANKLE: "left ankle",
    RIGHT_ANKLE: "right ankle",
}

# Skeleton edges used purely for drawing an overlay.
POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL),
    (LEFT_HEEL, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL),
    (RIGHT_HEEL, RIGHT_FOOT_INDEX),
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
]


@dataclass(frozen=True)
class Joint:
    """A three-point angle: the angle at ``vertex`` formed by ``a`` and ``c``.

    ``increase`` / ``decrease`` are the coaching cues to give when the athlete's
    angle is, respectively, *smaller* than the reference (open it up) or
    *larger* than the reference (close it down). Phrasing them here keeps the
    feedback engine declarative.
    """

    name: str
    side: str  # "left", "right", or "center"
    a: int
    vertex: int
    c: int
    increase: str  # cue when the user's angle is too small
    decrease: str  # cue when the user's angle is too large


# The joints that define the "pose" of a movement. Angles are computed at the
# vertex. A larger elbow angle => straighter arm; larger knee angle => straighter
# leg; larger shoulder angle => arm lifted further from the torso; larger hip
# angle => a more open/upright hip (less flexion).
JOINTS: list[Joint] = [
    Joint(
        "left_elbow", "left", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST,
        increase="straighten your left arm",
        decrease="bend your left elbow more",
    ),
    Joint(
        "right_elbow", "right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
        increase="straighten your right arm",
        decrease="bend your right elbow more",
    ),
    Joint(
        "left_shoulder", "left", LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP,
        increase="raise your left arm higher / further from your body",
        decrease="lower your left arm / bring it closer to your body",
    ),
    Joint(
        "right_shoulder", "right", RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP,
        increase="raise your right arm higher / further from your body",
        decrease="lower your right arm / bring it closer to your body",
    ),
    Joint(
        "left_hip", "left", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE,
        increase="open your left hip / stand taller through the left side",
        decrease="hinge/bend more at your left hip",
    ),
    Joint(
        "right_hip", "right", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE,
        increase="open your right hip / stand taller through the right side",
        decrease="hinge/bend more at your right hip",
    ),
    Joint(
        "left_knee", "left", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
        increase="straighten your left leg",
        decrease="bend your left knee more",
    ),
    Joint(
        "right_knee", "right", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
        increase="straighten your right leg",
        decrease="bend your right knee more",
    ),
]

JOINTS_BY_NAME = {j.name: j for j in JOINTS}
JOINT_NAMES = [j.name for j in JOINTS]

# Landmarks used to build the torso frame for normalization.
TORSO_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
