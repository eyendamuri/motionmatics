# Motionmatics

**Give it an example clip and your attempt. It tells you, in plain language, what
to change to move like the example**: which joints to bend or straighten, which
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
                               Raise your arms higher. You're 1.4× too slow;
                               speed up on the way down."
```

## How it works

1. **Pose extraction** (`pose.py`): MediaPipe's `PoseLandmarker` (Tasks API)
   gives 33 3-D body landmarks per frame for each video.
2. **Joint angles** (`angles.py`): eight joint angles (both elbows, shoulders,
   hips, knees) are computed from the 3-D landmarks. Angles are invariant to
   camera position, body size, and where the person stands in frame, so two
   people doing the "same" move produce comparable curves.
3. **Time alignment** (`align.py`): DTW finds the best monotonic
   frame-to-frame correspondence, absorbing differences in speed and start time.
   Alignment runs on the *standardized shape* of each clip so it lines up by
   timing, and never hides a shallow squat by matching it to the reference's
   mid-depth frames.
4. **Heuristic coaching** (`feedback.py`): along the aligned path it measures
   each joint's signed error (direction) and magnitude (severity), weighting the
   moments of exertion more than idle standing. It emits ranked, plain-language
   corrections, a per-phase breakdown, a tempo analysis, and a 0-100 match score.
5. **Visualization** (`visualize.py`): an overlay video of you (solid) vs the
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

Try it with no video at all: a built-in synthetic demo (a squat + arm-raise rep
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
| `--video / --plot / --json` | write the overlay MP4 / angle plot / JSON report | - |
| `--advice` | also compose spoken-style coaching advice from the measurements | off |
| `--activity NAME` | name the movement (e.g. `squat`) so the advice can mention it | - |

### Words of advice (`--advice`)

The heuristic output is composed into the advice a coach would say out loud
by a small **data-to-text engine** (`motionmatics/advice.py`), not a language
model. It runs on any OS, needs no downloads or keys, and answers instantly.

```bash
python -m motionmatics compare you.mp4 reference.mp4 --advice --activity squat
```

```
Coach's advice:
You're getting there on your squat: 69 out of 100, so let's tighten a few
things. The biggest thing: raise both arms higher and further from your body,
off by about 28 degrees, most of all in the middle. Lastly, bend both knees
more, off by about 24 degrees, again in the middle. Timing-wise, you're taking
about 40% longer than the reference, and the start is where you lose it, so
pick up the pace there. Lock in the first fix and re-test; the rest will
follow.
```

**Why not an LLM?** The input space is bounded and fully structured (8 joints
x 2 directions x severity x phase x tempo), so generalization comes from
*composition*, not parameters: the classic NLG pipeline (content selection →
aggregation → lexicalization → realization). Bilateral faults merge ("left
arm" + "right arm" → *"raise both arms"*), severities pick their wording,
phases attach where they belong ("most of all in the middle… again in the
middle"), tempo ratios become percentages, and discourse connectives sequence
it all. Every clause traces to a report field, so nothing *can* hallucinate;
an experiment with a local 3B LLM produced fluent text but misread errors as
angles, looped, or invented details, while being ~40x slower and needing a
1.8 GB download. Phrasing varies between different reports (seeded by the
report's content) but is deterministic for the same report, so it's testable.

## Example output

```
Motion match: 69/100  (Fair, several things to fix)

Top corrections:
 1. Raise your left arm higher / further from your body, about 28° too bent/closed (worst middle).
 2. Bend your left knee more, about 24° too straight/open (worst middle).
 ...
Timing: your motion is 1.40× slower than the reference overall.
        You're most out of time in the start, so try to speed up there.

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
- Angles are orientation-robust but not miracle workers: a front-on video can't
  measure a purely front-to-back (sagittal) motion well, and vice-versa.

## Design notes / limitations

- Comparison is **angle-based**, so it is blind to absolute position in space and
  to left/right mirroring by design (that's usually what you want for coaching).
- MediaPipe estimates a single person; multi-person clips use the most prominent.
- Feedback is heuristic and explainable on purpose: thresholds live at the top
  of `feedback.py` (`ANGLE_TOLERANCE_DEG`, score bounds, activity weighting) and
  are easy to tune per sport.

## Tests

```bash
python tests/test_motionmatics.py     # or: pytest
```
