"""Rendering: a coaching overlay video and angle-comparison plots.

The overlay video draws your body (solid) with the time-aligned reference "ghost"
(dashed) on top, both centred and scaled to the same torso size so the shapes are
directly comparable. Joints that are off by more than the tolerance flash red,
and a caption shows the current top cue.
"""

from __future__ import annotations

import numpy as np

from .compare import ComparisonResult
from .feedback import ANGLE_TOLERANCE_DEG
from .landmarks import JOINTS, JOINT_NAMES, POSE_CONNECTIONS
from .normalize import normalize_world


def _ref_for_user(result: ComparisonResult) -> np.ndarray:
    """Map every user frame to a single aligned reference frame (mean of matches)."""
    ui, ri = result.alignment.user_idx, result.alignment.ref_idx
    Tu = result.user_seq.num_frames
    out = np.zeros(Tu, dtype=int)
    for i in range(Tu):
        matches = ri[ui == i]
        out[i] = int(round(matches.mean())) if len(matches) else (out[i - 1] if i else 0)
    return out


def _project(norm_xy: np.ndarray, w: int, h: int) -> np.ndarray:
    """Project normalized (torso-unit) world x/y into pixel coordinates.

    MediaPipe world space has +y pointing *down*, matching screen pixels, so we
    add (not subtract) y to keep "up in the world" at the top of the frame.
    """
    scale = 0.22 * h  # 1 torso-length ~= 22% of canvas height
    cx, cy = w // 2, int(h * 0.45)
    px = cx + norm_xy[:, 0] * scale
    py = cy + norm_xy[:, 1] * scale
    return np.stack([px, py], axis=1)


def _draw_skeleton(img, pts, color, thickness=3, dashed=False):
    import cv2
    for a, b in POSE_CONNECTIONS:
        pa, pb = tuple(pts[a].astype(int)), tuple(pts[b].astype(int))
        if dashed:
            # draw a dashed line
            dist = int(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
            n = max(dist // 10, 1)
            for k in range(0, n, 2):
                t0, t1 = k / n, min((k + 1) / n, 1.0)
                p0 = (int(pa[0] + (pb[0] - pa[0]) * t0), int(pa[1] + (pb[1] - pa[1]) * t0))
                p1 = (int(pa[0] + (pb[0] - pa[0]) * t1), int(pa[1] + (pb[1] - pa[1]) * t1))
                cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        else:
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def render_overlay_video(
    result: ComparisonResult,
    out_path: str,
    size: tuple[int, int] = (900, 700),
    fps: float | None = None,
) -> str:
    """Write an MP4 of the user (solid) vs aligned reference ghost (dashed)."""
    import cv2

    w, h = size
    fps = fps or result.user_seq.fps or 30.0
    ref_for_user = _ref_for_user(result)

    user_norm = normalize_world(result.user_seq)
    ref_norm = normalize_world(result.ref_seq)

    # per-frame per-joint |error| for colouring
    ui, ri = result.alignment.user_idx, result.alignment.ref_idx

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    top_cue = result.report.corrections[0].cue if result.report.corrections else "great form!"

    for i in range(result.user_seq.num_frames):
        j = ref_for_user[i]
        canvas = np.full((h, w, 3), 22, dtype=np.uint8)

        upts = _project(user_norm[i, :, :2], w, h)
        rpts = _project(ref_norm[j, :, :2], w, h)

        _draw_skeleton(canvas, rpts, (120, 120, 120), thickness=2, dashed=True)   # ghost
        _draw_skeleton(canvas, upts, (90, 220, 90), thickness=4)                  # you

        # highlight off joints
        for jd in JOINTS:
            ua = _angle(user_norm[i], jd)
            ra = _angle(ref_norm[j], jd)
            if not np.isnan(ua) and not np.isnan(ra) and abs(ua - ra) > ANGLE_TOLERANCE_DEG:
                c = tuple(int(x) for x in upts[jd.vertex].astype(int))
                cv2.circle(canvas, c, 10, (60, 60, 235), -1, cv2.LINE_AA)

        cv2.putText(canvas, "you", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 220, 90), 2)
        cv2.putText(canvas, "reference (ghost)", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 2)
        cv2.putText(canvas, f"Fix: {top_cue}", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 235), 2)
        cv2.putText(canvas, f"score {result.report.score:.0f}/100", (w - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        writer.write(canvas)

    writer.release()
    return out_path


def _angle(frame_landmarks: np.ndarray, joint) -> float:
    a, b, c = frame_landmarks[joint.a], frame_landmarks[joint.vertex], frame_landmarks[joint.c]
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / (na * nc), -1, 1)
    return float(np.degrees(np.arccos(cos)))


def plot_angle_comparison(result: ComparisonResult, out_path: str) -> str:
    """Save a PNG grid of per-joint angle curves: you vs time-aligned reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_for_user = _ref_for_user(result)
    t = result.user_seq.timestamps()
    ref_on_user = result.ref_angles[ref_for_user]

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for k, name in enumerate(JOINT_NAMES):
        ax = axes[k // 2, k % 2]
        ax.plot(t, result.user_angles[:, k], color="#2ca02c", label="you", lw=2)
        ax.plot(t, ref_on_user[:, k], color="#888", ls="--", label="reference", lw=2)
        ax.fill_between(t, result.user_angles[:, k], ref_on_user[:, k],
                        where=np.abs(result.user_angles[:, k] - ref_on_user[:, k]) > ANGLE_TOLERANCE_DEG,
                        color="#d62728", alpha=0.2)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_ylabel("deg")
        if k == 0:
            ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(
        f"Motionmatics: angle comparison (match {result.report.score:.0f}/100)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
