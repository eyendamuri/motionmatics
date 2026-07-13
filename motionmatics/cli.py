"""Command-line interface.

    python -m motionmatics compare you.mp4 reference.mp4 --video out.mp4 --plot out.png
    python -m motionmatics demo --outdir demo_out
    python -m motionmatics extract clip.mp4 poses.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_compare_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="path to pose_landmarker .task model")
    p.add_argument("--phases", type=int, default=3, help="number of movement phases (default 3)")
    p.add_argument("--smooth", type=int, default=5, help="angle smoothing window (frames)")
    p.add_argument("--band", type=float, default=0.2, help="DTW Sakoe-Chiba band fraction")
    p.add_argument("--max-frames", type=int, default=None, help="cap frames per clip")
    p.add_argument("--json", dest="json_out", default=None, help="write report JSON here")
    p.add_argument("--video", dest="video_out", default=None, help="write overlay MP4 here")
    p.add_argument("--plot", dest="plot_out", default=None, help="write angle-plot PNG here")
    p.add_argument("--ai", action="store_true",
                   help="also generate natural-language coaching advice with Claude")
    p.add_argument("--activity", default=None,
                   help="name of the movement (e.g. 'squat') to give the AI coach context")


def cmd_compare(args) -> int:
    from .compare import compare_videos

    for v in (args.user, args.reference):
        if not os.path.exists(v):
            print(f"error: file not found: {v}", file=sys.stderr)
            return 2

    result = compare_videos(
        user_video=args.user,
        ref_video=args.reference,
        model_path=args.model,
        n_phases=args.phases,
        smooth=args.smooth,
        band=args.band,
        max_frames=args.max_frames,
    )
    _emit(result, args)
    return 0


def cmd_demo(args) -> int:
    from .compare import compare_sequences
    from .synthetic import demo_pair

    ref, user = demo_pair()
    result = compare_sequences(user, ref, n_phases=args.phases, smooth=args.smooth, band=args.band)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        args.json_out = args.json_out or os.path.join(args.outdir, "report.json")
        args.video_out = args.video_out or os.path.join(args.outdir, "overlay.mp4")
        args.plot_out = args.plot_out or os.path.join(args.outdir, "angles.png")
    _emit(result, args)
    return 0


def cmd_extract(args) -> int:
    from .pose import PoseExtractor

    extractor = PoseExtractor(model_path=args.model)
    seq = extractor.extract(args.video, label=args.label, progress=True)
    seq.save(args.out)
    print(f"saved {seq.num_frames} frames ({seq.duration:.1f}s) -> {args.out}")
    return 0


def _emit(result, args) -> None:
    print()
    print(result.report.render_text())
    print()
    if getattr(args, "ai", False):
        from .ai_coach import AICoachError, ai_advice

        try:
            advice = ai_advice(result.report, activity=getattr(args, "activity", None))
            print("Coach's advice:")
            print(advice)
            print()
        except AICoachError as e:
            print(f"[ai] skipped: {e}", file=sys.stderr)
    if getattr(args, "json_out", None):
        with open(args.json_out, "w") as f:
            json.dump(result.report.to_dict(), f, indent=2)
        print(f"[json]  {args.json_out}")
    if getattr(args, "video_out", None):
        from .visualize import render_overlay_video
        render_overlay_video(result, args.video_out)
        print(f"[video] {args.video_out}")
    if getattr(args, "plot_out", None):
        from .visualize import plot_angle_comparison
        plot_angle_comparison(result, args.plot_out)
        print(f"[plot]  {args.plot_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="motionmatics", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compare", help="compare your video to a reference video")
    c.add_argument("user", help="your attempt (video)")
    c.add_argument("reference", help="the example/target (video)")
    _add_compare_args(c)
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("demo", help="run the built-in synthetic demo (no video needed)")
    d.add_argument("--outdir", default=None, help="write report.json, overlay.mp4, angles.png here")
    _add_compare_args(d)
    d.set_defaults(func=cmd_demo)

    e = sub.add_parser("extract", help="extract poses from a video to .npz")
    e.add_argument("video")
    e.add_argument("out")
    e.add_argument("--model", default=None)
    e.add_argument("--label", default="pose")
    e.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
