# Padel Game Analytics — Shot Classification System

A computer vision prototype that analyzes padel match footage to detect players, ball, and rackets, classify shot types, and generate match analytics.

---

## How It Works

```
Video → Detect (YOLO + OpenCV) → Track → Classify Shots → Save Results + Dashboard
```

Each component is a separate module — easy to swap or improve independently.

---

## Key Challenges & How I Solved Them

**Ball detection is hard.**
YOLO's pretrained model misses small fast padel balls. I added two OpenCV fallbacks: HSV color segmentation (yellow-green range) and background subtraction (MOG2). If YOLO misses, one of the fallbacks usually catches it.

**No labeled shot data exists.**
Instead of training a model, I used rule-based classification on ball trajectory:
- Smash → fast downward motion
- Serve → upward arc + high speed
- Forehand/Backhand → ball position relative to nearest player's center
- Rally → moderate speed with 2+ players visible

**Racket detection without retraining.**
COCO class 38 (tennis racket) is close enough to a padel racket. I raised the confidence threshold to 0.45 to reduce false positives.

**Bounce detection.**
A bounce happens when the ball's vertical direction flips from downward to upward. Simple rule, works well with smoothed trajectory data.

---

## Tech Stack

Python · YOLOv8 · OpenCV · ByteTrack · Pandas · Matplotlib

---

## Project Structure

```
src/
├── detector.py        # YOLO + HSV + motion fallback for ball, racket detection
├── tracker.py         # Ball tracking (EMA smoothing, velocity) + bounce detection
├── shot_classifier.py # Rule-based shot classification + player assignment
├── analytics.py       # Export JSON/CSV + generate dashboard
├── visualizer.py      # Draw boxes, ball circle, shot label on video
└── pipeline.py        # Wires everything together

data/                  # Input video
models/                # YOLOv8 weights
results/               # All outputs
```

---

## Outputs

| File | Contents |
|---|---|
| `shots_detected.json` | frame, timestamp, shot type, player ID |
| `shots.csv` | same as above in CSV |
| `summary.json` | total shots, bounces, per-type counts |
| `ball_trajectory.json` | ball position + velocity per frame |
| `racket_tracking.json` | racket positions per track ID |
| `bounces.json` | bounce events with frame and position |
| `dashboard.png` | 5-chart visual summary |

**shots_detected.json example:**
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

5 charts in one image:
- Bar chart — shot count by type
- Pie chart — shot type percentages
- Scatter plot — when each shot happened in the match
- Line chart — players and rackets detected per frame, bounce markers
- Summary panel — totals and per-type breakdown

---

## Limitations (honest)

- Ball detection still fails during occlusion or fast motion
- Forehand/backhand logic assumes right-handed players
- Racket detection is noisy — COCO class 38 is not trained on padel rackets
- No rally segmentation or court boundary detection

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Results saved to `results/`.
