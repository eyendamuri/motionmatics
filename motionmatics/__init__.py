"""Motionmatics: heuristic motion coaching by comparing two videos.

Give it an example clip and your attempt; it extracts body poses, time-aligns the
two performances, and tells you in plain language what to change to match the
example (which joints to bend/straighten, and whether to speed up or slow down).
"""

from .align import Alignment, align_angles, dtw
from .angles import angle_matrix, joint_angles, joint_visibility
from .compare import ComparisonResult, compare_sequences, compare_videos
from .feedback import FeedbackReport, generate_feedback
from .pose import PoseExtractor, PoseSequence

__version__ = "0.1.0"

__all__ = [
    "PoseSequence",
    "PoseExtractor",
    "joint_angles",
    "angle_matrix",
    "joint_visibility",
    "align_angles",
    "dtw",
    "Alignment",
    "generate_feedback",
    "FeedbackReport",
    "compare_sequences",
    "compare_videos",
    "ComparisonResult",
]
