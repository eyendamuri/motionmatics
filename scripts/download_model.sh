#!/usr/bin/env bash
# Download the MediaPipe Pose Landmarker model used by Motionmatics.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
echo "Downloading pose_landmarker_full.task ..."
curl -sSL -o models/pose_landmarker_full.task "$URL"
echo "Saved to models/pose_landmarker_full.task"
