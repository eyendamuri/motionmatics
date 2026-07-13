# Motionmatics

**Give it an example clip and your attempt. It tells you, in plain language, what
to change to move like the example** — which joints to bend or straighten, which
arm to raise, and whether to speed up or slow down.

No machine-learning training, no black box. Motionmatics extracts body poses with
MediaPipe, time-aligns the two performances with Dynamic Time Warping, and runs a
transparent, rule-based heuristic over the joint angles. Every cue traces back to
a measured difference in degrees at a specific joint and moment.

```
reference video ─┐
                 ├─► poses ─► joint angles ─► DTW time-align ─► heuristic coach
your video ──────┘                                                   │
                                                                     ▼
                              "Bend your knees ~24° more (worst at the bottom).
                               Raise your arms higher. You're 1.4× too slow —
                               speed up on the way down."
```

## How it works

1. **Pose extraction** (`pose.py`) — MediaPipe's `PoseLandmarker` (Tasks API)
   gives 33 3-D body landmarks per frame for each video.
2. **Joint angles** (`angles.py`) — eight joint angles (both elbows, shoulders,
   hips, knees) are computed from the 3-D landmarks. Angles are invariant to
   camera position, body size, and where the person stands in frame, so two
   people doing the "same" move produce comparable curves.
3. **Time alignment** (`align.py`) — DTW finds the best monotonic
   frame-to-frame correspondence, absorbing differences in speed and start time.
   Alignment runs on the *standardized shape* of each clip so it lines up by
   timing, and never hides a shallow squat by matching it to the reference's
   mid-depth frames.
4. **Heuristic coaching** (`feedback.py`) — along the aligned path it measures
   each joint's signed error (direction) and magnitude (severity), weighting the
   moments of exertion more than idle standing. It emits ranked, plain-language
   corrections, a per-phase breakdown, a tempo analysis, and a 0–100 match score.
5. **Visualization** (`visualize.py`) — an overlay video of you (solid) vs the
   time-aligned reference "ghost" (dashed) with off-joints flashing red, plus a
   per-joint angle-comparison plot.

## Install

```bash
conda create -n motionmatics python=3.11 -y && conda activate motionmatics
pip install -r requirements.txt          # or: pip install -e .
bash scripts/download_model.sh           # fetches the 9 MB pose model
```

MediaPipe currently ships wheels for Python ≤ 3.12; use 3.11 to be safe.

## Usage

Compare two videos of the same movement (yours first, the example second):

```bash
python -m motionmatics compare you.mp4 reference.mp4 \
    --video overlay.mp4 --plot angles.png --json report.json
```

Try it with no video at all — a built-in synthetic demo (a squat + arm-raise rep
where the "user" squats shallow, under-raises the arms, and is 40% too slow):

```bash
python -m motionmatics demo --outdir demo_out
```

Extract poses once and reuse them:

```bash
python -m motionmatics extract clip.mp4 clip_poses.npz
```

### Options

| flag | meaning | default |
|------|---------|---------|
| `--phases N` | split the movement into N phases for the breakdown | 3 |
| `--smooth W` | moving-average window (frames) to de-jitter angles | 5 |
| `--band F`   | DTW Sakoe-Chiba band as a fraction of the longer clip | 0.2 |
| `--max-frames N` | cap frames per clip (speed) | none |
| `--video / --plot / --json` | write the overlay MP4 / angle plot / JSON report | — |
| `--ai` | also generate natural-language coaching advice with Claude | off |
| `--activity NAME` | tell the AI coach what the movement is (e.g. `squat`) | — |

### AI coach (`--ai`)

The heuristic engine's structured output (per-joint errors in degrees, phases,
tempo ratios) is handed to **Claude** (`claude-opus-4-8`, adaptive thinking via
the official `anthropic` SDK), which turns it into the advice a coach would say
out loud — prioritized, plain-language, encouraging. The model is instructed to
use *only* the measured numbers, so the facts stay grounded in the deterministic
pipeline; the LLM contributes the phrasing.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or: ant auth login
python -m motionmatics compare you.mp4 reference.mp4 --ai --activity squat
```

Without credentials the report still prints in full and the AI step is skipped
with a one-line hint (see `motionmatics/ai_coach.py`).

## Example output

```
Motion match: 69/100  (Fair — several things to fix)

Top corrections:
 1. Raise your left arm higher / further from your body — about 28° too bent/closed (worst middle).
 2. Bend your left knee more — about 24° too straight/open (worst middle).
 ...
Timing: your motion is 1.40× slower than the reference overall.
        You're most out of time in the start — try to speed up there.

Phase-by-phase:
 - Start: raise your right arm higher (≈16° off).
 - Middle: raise your left arm higher (≈37° off).
 - End: looking good.
```

## Tips for good results

- Film both clips from a **similar angle** (front-on or side-on) and keep the
  whole body in frame.
- One clean **repetition** per clip works best; the phase breakdown assumes a
  single movement, not a long montage.
- Angles are orientation-robust but not miracle workers — a front-on video can't
  measure a purely front-to-back (sagittal) motion well, and vice-versa.

## Design notes / limitations

- Comparison is **angle-based**, so it is blind to absolute position in space and
  to left/right mirroring by design (that's usually what you want for coaching).
- MediaPipe estimates a single person; multi-person clips use the most prominent.
- Feedback is heuristic and explainable on purpose — thresholds live at the top
  of `feedback.py` (`ANGLE_TOLERANCE_DEG`, score bounds, activity weighting) and
  are easy to tune per sport.

## Tests

```bash
python tests/test_motionmatics.py     # or: pytest
```
