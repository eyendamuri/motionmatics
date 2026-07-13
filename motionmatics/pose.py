"""Pose extraction.

``PoseSequence`` is the in-memory representation of a tracked body over time.
``PoseExtractor`` wraps MediaPipe's Tasks ``PoseLandmarker`` (the modern API;
the legacy ``mp.solutions`` module is gone in mediapipe >= 0.10.18) and turns a
video file into a ``PoseSequence``.

The extractor is optional: everything downstream operates on ``PoseSequence``
arrays, so poses can also be loaded from ``.npz`` files or generated
synthetically (see :mod:`motionmatics.synthetic`) without MediaPipe installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .landmarks import NUM_LANDMARKS

# Default location of the bundled model, resolved relative to the repo root.
_DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"


@dataclass
class PoseSequence:
    """A tracked pose over ``T`` frames.

    Attributes
    ----------
    world:
        ``(T, 33, 3)`` metric landmarks in MediaPipe world space (origin at the
        mid-hip, in metres). Angles are computed from these because they are
        camera- and scale-robust.
    image:
        ``(T, 33, 3)`` image-normalized landmarks, ``x``/``y`` in ``[0, 1]``
        relative to frame size (``z`` relative). Used for drawing overlays.
    visibility:
        ``(T, 33)`` per-landmark visibility in ``[0, 1]``.
    fps:
        Source frame rate (frames per second).
    label:
        Human-friendly name (e.g. "reference" or "you").
    """

    world: np.ndarray
    image: np.ndarray
    visibility: np.ndarray
    fps: float
    label: str = "pose"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.world = np.asarray(self.world, dtype=np.float64)
        self.image = np.asarray(self.image, dtype=np.float64)
        self.visibility = np.asarray(self.visibility, dtype=np.float64)
        if self.world.shape[1:] != (NUM_LANDMARKS, 3):
            raise ValueError(f"world must be (T,{NUM_LANDMARKS},3), got {self.world.shape}")
        if self.image.shape[1:] != (NUM_LANDMARKS, 3):
            raise ValueError(f"image must be (T,{NUM_LANDMARKS},3), got {self.image.shape}")

    @property
    def num_frames(self) -> int:
        return self.world.shape[0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0

    def timestamps(self) -> np.ndarray:
        return np.arange(self.num_frames) / (self.fps or 1.0)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | os.PathLike) -> None:
        np.savez_compressed(
            path,
            world=self.world,
            image=self.image,
            visibility=self.visibility,
            fps=np.array([self.fps]),
            label=np.array([self.label]),
            meta=np.array([json.dumps(self.meta)]),
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "PoseSequence":
        d = np.load(path, allow_pickle=False)
        meta = {}
        if "meta" in d:
            try:
                meta = json.loads(str(d["meta"][0]))
            except Exception:
                meta = {}
        return cls(
            world=d["world"],
            image=d["image"],
            visibility=d["visibility"],
            fps=float(d["fps"][0]),
            label=str(d["label"][0]) if "label" in d else "pose",
            meta=meta,
        )


class PoseExtractor:
    """Extract a :class:`PoseSequence` from a video using MediaPipe Tasks."""

    def __init__(self, model_path: str | os.PathLike | None = None, num_poses: int = 1):
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {self.model_path}. Download it with:\n"
                "  curl -sSL -o models/pose_landmarker_full.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )
        self.num_poses = num_poses

    def extract(
        self,
        video_path: str | os.PathLike,
        label: str = "pose",
        max_frames: int | None = None,
        progress: bool = False,
    ) -> PoseSequence:
        import cv2  # local import so the package imports without cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        options = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        world_frames: list[np.ndarray] = []
        image_frames: list[np.ndarray] = []
        vis_frames: list[np.ndarray] = []

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                ts_ms = int(1000 * frame_idx / fps)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                world, image, vis = _result_to_arrays(result)
                world_frames.append(world)
                image_frames.append(image)
                vis_frames.append(vis)

                frame_idx += 1
                if progress and frame_idx % 30 == 0:
                    print(f"  [{label}] processed {frame_idx} frames", flush=True)
                if max_frames and frame_idx >= max_frames:
                    break
        cap.release()

        if not world_frames:
            raise RuntimeError(f"No frames read from {video_path}")

        seq = PoseSequence(
            world=np.stack(world_frames),
            image=np.stack(image_frames),
            visibility=np.stack(vis_frames),
            fps=float(fps),
            label=label,
            meta={"source": os.path.basename(video_path)},
        )
        return _interpolate_missing(seq)


def _result_to_arrays(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one PoseLandmarkerResult into (world, image, visibility) arrays.

    Missing detections become NaN so we can interpolate later.
    """
    world = np.full((NUM_LANDMARKS, 3), np.nan)
    image = np.full((NUM_LANDMARKS, 3), np.nan)
    vis = np.zeros(NUM_LANDMARKS)

    if result.pose_landmarks and result.pose_world_landmarks:
        img_lm = result.pose_landmarks[0]
        wl = result.pose_world_landmarks[0]
        for i in range(NUM_LANDMARKS):
            image[i] = (img_lm[i].x, img_lm[i].y, img_lm[i].z)
            world[i] = (wl[i].x, wl[i].y, wl[i].z)
            vis[i] = getattr(img_lm[i], "visibility", 1.0)
    return world, image, vis


def _interpolate_missing(seq: PoseSequence) -> PoseSequence:
    """Fill short gaps of undetected frames by linear interpolation."""
    for arr in (seq.world, seq.image):
        T, J, C = arr.shape
        for j in range(J):
            for c in range(C):
                col = arr[:, j, c]
                mask = np.isnan(col)
                if mask.all():
                    col[:] = 0.0
                elif mask.any():
                    idx = np.arange(T)
                    col[mask] = np.interp(idx[mask], idx[~mask], col[~mask])
    seq.visibility = np.nan_to_num(seq.visibility, nan=0.0)
    return seq
