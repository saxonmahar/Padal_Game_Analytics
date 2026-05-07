# Padel Game Analytics — Shot Classification System

This project shows how I solve problems when the "right" approach isn't available. No padel training data existed, so instead of forcing a model that would fail, I built a hybrid detection system, implemented proper tracking, and wrote rule-based classification with enough context to be meaningful.

---

## What It Does

- Detects the ball using a three-layer hybrid approach — YOLO, HSV color segmentation, background subtraction
- Tracks players and rackets with persistent IDs across the full video using ByteTrack
- Classifies shots — forehand, backhand, smash, serve, rally — using trajectory analysis with spatial and temporal context
- Assigns each shot to the nearest player using tracked IDs
- Detects ball bounces using vertical velocity reversal
- Draws a fading ball trail on the video to visualize trajectory quality
- Exports structured results as JSON and CSV
- Generates a 5-chart analytics dashboard
- Measures accuracy against 25 manually labeled ground truth shots

---

## How to Run

```bash
pip install -r requirements.txt

# Run the pipeline
python main.py

# Evaluate accuracy against ground truth
python evaluate.py
```

Results saved to `results/`.

---

## Architecture

```
Video Input
    ↓
Detector   →  YOLO (players, rackets) + HSV + Motion fallback (ball)
    ↓
Tracker    →  EMA smoothing · missing frame interpolation · bounce detection
    ↓
Classifier →  Rules: player-relative height + player-relative direction + bounce timing
    ↓
Analytics  →  JSON · CSV · summary stats · dashboard PNG
    ↓
Visualizer →  Ball trail · bounding boxes · shot labels on video
```

---

## Key Technical Decisions

### Ball Detection — Hybrid Over Single Model

I identified early that YOLO's pretrained COCO model couldn't detect padel balls — under 10% accuracy on the test video. The model was trained on soccer and basketballs, not a 15-pixel fast-moving object.

Instead of forcing it, I built a three-layer cascade:
1. **YOLO (class 32)** — runs first, works when the ball is visible and large enough
2. **HSV color segmentation** — thresholds the yellow-green color range, scores contours by circularity
3. **Background subtraction (MOG2)** — highlights anything moving against the static court, picks small circular blobs

Result: detection rate went from ~10% to ~60–70%. The visualizer color-codes which method fired per frame — cyan for YOLO, orange for HSV, green for motion. That debugging tool alone saved hours of guessing.

---

### Player Tracking — ByteTrack for Persistent IDs

Without persistent tracking, player IDs reset every few frames and shot assignment becomes meaningless. Switching from `model()` to `model.track(persist=True)` was a small code change with a large impact — player 2 in frame 100 is the same player 2 in frame 500. Shot-to-player assignment now means something.

---

### Shot Classification — Rules With Context, Not Naive Thresholds

No labeled padel shot dataset exists. I looked at tennis models but the mechanics are different enough that transfer learning wasn't reliable. MediaPipe pose estimation ran at 15fps on my hardware and struggled with the overhead camera angle — I chose reliability over complexity.

I built rule-based classification, but with three layers of context that make it less naive:

**Player-relative height** — instead of raw pixel dy (which is camera-angle dependent), I compute where the ball is relative to the nearest player's bounding box. Above the head = smash territory. Waist height = forehand/backhand. This removes the camera angle bias entirely.

**Player-relative direction** — instead of raw dx, I compute whether the ball is to the left or right of the nearest player's center. Camera-angle independent forehand/backhand classification.

**Bounce timing for serve** — a serve only fires if there was a bounce in the last 45 frames. In padel, a serve always follows the player bouncing the ball. Without this, any fast upward ball movement gets called a serve. This catches a key padel rule that a naive classifier misses.

| Shot | Conditions |
|---|---|
| Smash | Ball above player head + fast speed + near player/racket |
| Serve | Ball at waist/low + upward motion + recent bounce |
| Forehand | Ball at waist height + ball right of player center |
| Backhand | Ball at waist height + ball left of player center |
| Rally | Moderate speed + 2+ players visible + ball near someone |

---

### Evaluation — Measuring Instead of Guessing

I built `evaluate.py` against 25 manually labeled shots. That's a small sample, but having an actual number tells me where to focus next.

```bash
python evaluate.py
```

```
Detection rate : 80.0%   (shot found near the right frame)
Accuracy       : 64.0%   (correct type / total ground truth)

Per-class:
  forehand    80.0%
  backhand    66.7%
  smash       75.0%
  serve       60.0%
  rally       50.0%
```

The evaluation identified rally and serve as the weak spots. My next iteration would focus there — tighter bounce detection for serve, and a minimum speed threshold to separate rally from noise.

Full report: `results/evaluation_report.json`

---

### Why Not MediaPipe Pose Estimation

I considered it. Pose estimation would give wrist positions, elbow angles, and shoulder rotation — the right features for reliable forehand/backhand classification.

I decided against it for three reasons:

1. **Speed** — MediaPipe runs at ~15fps on CPU on my machine. The pipeline already runs slower than real time. Adding pose estimation would make it significantly worse.

2. **Camera angle** — the test video is shot from a high overhead position. Pose estimation works best from side or front views. From above, shoulder rotation and arm angles are foreshortened and unreliable. The model was trained mostly on upright front-facing poses.

3. **Risk at this stage** — adding a new dependency that might fail or slow things down is a bad trade. I chose reliability over complexity.

What I did instead: made the existing features camera-angle independent through player-relative height and direction. Same root problem, no new dependencies.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| YOLOv8 (Ultralytics) | Object detection |
| ByteTrack | Multi-object tracking with persistent IDs |
| OpenCV | HSV filtering, background subtraction, video I/O |
| NumPy | Velocity and distance calculations |
| Pandas | CSV export |
| Matplotlib | Dashboard generation |

---

## Project Structure

```
src/
├── detector.py        # Hybrid ball detection + racket detection
├── tracker.py         # Ball tracking, EMA smoothing, bounce detection
├── shot_classifier.py # Shot classification + player assignment
├── analytics.py       # Data export + dashboard
├── visualizer.py      # Video overlay — ball trail, boxes, shot labels
└── pipeline.py        # Main processing loop

data/
└── ground_truth.json  # 25 manually labeled shots for evaluation

evaluate.py            # Accuracy evaluation script
main.py                # Entry point
```

---

## Output Files

| File | Contents |
|---|---|
| `shots_detected.json` | Frame, timestamp, shot type, player ID |
| `shots.csv` | Same in CSV |
| `summary.json` | Totals and per-type breakdown |
| `ball_trajectory.json` | Ball position + velocity per frame |
| `racket_tracking.json` | Racket positions per track ID |
| `bounces.json` | Bounce events with frame and position |
| `evaluation_report.json` | Accuracy vs ground truth |
| `dashboard.png` | 5-chart visual summary |

```json
{
  "frame_id": 384,
  "timestamp_seconds": 12.8,
  "shot_type": "forehand",
  "player_id": 2
}
```

---

## Dashboard

![Dashboard](assets/dashboard.png)

---

## Limitations

Ball detection still misses 30–40% of frames, especially when the ball goes behind a player or the net. The shot rules are tuned to one camera angle — a different video would need retuning. Racket detection uses COCO class 38 (tennis racket) which is noisy on padel footage. Forehand/backhand assumes right-handed players.

The 64% accuracy is on 25 samples — honest but not statistically strong. A proper evaluation needs more samples across multiple videos.

---

## If I Were Building This for Production

- Fine-tune YOLO on a padel-specific dataset — the single biggest lever for improvement
- Replace EMA smoothing with Kalman filtering for proper occlusion handling
- Add court homography to map pixel coordinates to real court coordinates — makes all spatial features truly camera-angle independent
- Add MediaPipe pose estimation on every 3rd frame with fallback to current rules
- Train a temporal classifier (SlowFast / I3D) on labeled shot clips to replace the rules
- Treat 64% as a baseline and improve through iterative labeling of edge cases
- Build the evaluation framework first — it would have helped tune thresholds much faster
