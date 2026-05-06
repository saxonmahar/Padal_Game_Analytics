# Padel Game Analytics — Shot Classification System

A computer vision prototype that analyzes padel match footage to detect players, ball, and rackets, classify shot types, and generate match analytics.

---

## How It Works

```
Video → Detect (YOLO + OpenCV) → Track → Classify Shots → Save Results + Dashboard
```

Each component is a separate module — easy to swap or improve independently.

---

## Challenges I Faced & How I Solved Them

### Challenge 1: Ball Detection Barely Works

**The Problem:**
YOLO's pretrained COCO model has a "sports ball" class (32), but it's trained on soccer balls and basketballs — large, slow-moving objects. A padel ball is tiny (maybe 10-20 pixels), moves extremely fast, and often gets motion-blurred. YOLO detected the ball in less than 10% of frames.

**What I Tried:**
- Lowering YOLO's confidence threshold → got more false positives (detecting random circular objects) but still missed most real balls
- Using YOLOv8x (larger model) → slightly better but still under 20% detection rate, and much slower

**How I Solved It:**
Added two OpenCV fallback methods that only run when YOLO fails:
1. **HSV color segmentation** — padel balls are yellow-green. I convert the frame to HSV color space, threshold for that color range, find contours, and pick the most circular one in the right size range.
2. **Background subtraction (MOG2)** — builds a model of the static background over time. Anything moving gets highlighted. Small circular moving blobs are likely the ball.

The three-layer cascade (YOLO → HSV → motion) brought detection rate up to around 60-70%. Still not perfect, but usable.

**What Still Doesn't Work:**
- When the ball passes behind a player or net (occlusion), all three methods fail
- HSV breaks under poor lighting or when the court has yellow lines
- Motion detection triggers on player hands or racket swings
- Fast camera pans confuse the background subtractor

**How to Improve:**
- Fine-tune YOLO on a padel-specific dataset with small ball annotations
- Use optical flow to predict ball position during occlusion
- Train a dedicated small-object detector (like YOLO-tiny with anchor boxes tuned for 10-20px objects)

---

### Challenge 2: No Labeled Data for Shot Classification

**The Problem:**
To train a deep learning shot classifier, I'd need hundreds of labeled video clips showing forehand, backhand, smash, serve, etc. No such dataset exists for padel. Creating one manually would take weeks.

**What I Tried:**
- Looking for tennis shot datasets to transfer learn from → found some, but tennis shots are mechanically different from padel (different court, different racket, different ball physics)
- Trying to use pose estimation (MediaPipe) to detect player body orientation → too slow (15 fps on my machine), and pose detection failed when players were far from camera or partially occluded

**How I Solved It:**
Used rule-based classification on ball trajectory features:
- Smash → fast downward motion (avg_dy > 18, speed > 20)
- Serve → upward arc + high speed (avg_dy < -10, speed > 15)
- Forehand/Backhand → ball position relative to nearest player's center (left side = backhand, right side = forehand)
- Rally → moderate speed with 2+ players visible

Rules are brittle but interpretable and require zero training data.

**What Still Doesn't Work:**
- Thresholds (like `avg_dy > 18`) are tuned to one video and break on others
- Forehand/backhand logic assumes right-handed players — left-handed players get misclassified
- Can't distinguish between a lob and a regular rally shot (both have similar speed/direction)
- No temporal context — each shot is classified independently, so the system doesn't know if it's the start of a rally or the end

**How to Improve:**
- Collect 200-300 labeled shot clips and train a temporal CNN (like SlowFast or I3D)
- Use pose estimation to detect racket swing direction instead of just ball position
- Add rally segmentation — group shots into points using serve detection + bounce patterns
- Use a sliding window classifier that looks at 1-2 seconds of trajectory instead of just 5 frames

---

### Challenge 3: Racket Detection Without Retraining

**The Problem:**
COCO has no padel racket class. The options were:
1. Fine-tune YOLO on a padel racket dataset (requires labeling 500+ images)
2. Use COCO class 38 (tennis racket) and hope it's close enough
3. Build a custom racket detector from scratch

**How I Solved It:**
Went with option 2 — COCO class 38. Tennis rackets and padel rackets are visually similar (both have a handle and a flat hitting surface). I raised the confidence threshold from 0.25 to 0.45 to reduce false positives.

**What Still Doesn't Work:**
- Detection is noisy — confidence scores fluctuate wildly frame to frame
- Rackets held at certain angles (edge-on to camera) don't get detected
- Sometimes detects player hands or the net as rackets
- No way to associate which player is holding which racket (ByteTrack gives racket IDs, but they don't map to player IDs)

**How to Improve:**
- Fine-tune YOLO on 300-500 labeled padel racket images
- Use pose estimation to detect wrist keypoints, then look for rackets near wrists
- Add temporal consistency — if a racket was detected at position X in frame N, it should be near X in frame N+1

---

### Challenge 4: Tracking Through Occlusion

**The Problem:**
When the ball passes behind a player or the net, detection fails for several frames. Without handling this, the trajectory gets broken into disconnected segments.

**How I Solved It:**
Added missing-frame interpolation in the tracker. If the ball isn't detected for up to 8 consecutive frames, the tracker holds the last known position instead of dropping it. This bridges short occlusions.

Also added EMA smoothing (alpha=0.6) to reduce jitter from noisy detections.

**What Still Doesn't Work:**
- If occlusion lasts more than 8 frames, the trajectory breaks
- Holding the last position is a naive interpolation — the ball is actually moving during occlusion, so the held position is wrong
- EMA smoothing introduces lag — fast direction changes (like a bounce) take 2-3 frames to register

**How to Improve:**
- Use Kalman filtering to predict ball position during occlusion based on velocity
- Use optical flow to track the ball even when it's not detected
- Reduce EMA alpha during high-speed motion to reduce lag

---

### Challenge 5: Assigning Shots to Players

**The Problem:**
Which player hit each shot? The system detects shots and detects players, but doesn't know which player caused which shot.

**How I Solved It:**
For each detected shot, find the player bounding box closest to the ball at that frame. Use ByteTrack IDs to maintain consistent player identity across frames.

**What Still Doesn't Work:**
- Nearest-player heuristic fails when players are close together or overlapping
- Doesn't account for racket position — a player can be near the ball but not actually hitting it
- No validation — if the ball is mid-court and both players are far away, the system still assigns it to the nearest one

**How to Improve:**
- Detect racket-ball contact events (when racket bounding box overlaps ball position)
- Use pose estimation to detect swing motion, then assign shots to players who are swinging
- Add a distance threshold — if no player is within X pixels of the ball, mark the shot as "unknown player"

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

![Dashboard](results/dashboard.png)

5 charts in one image:
- Bar chart — shot count by type
- Pie chart — shot type percentages
- Scatter plot — when each shot happened in the match
- Line chart — players and rackets detected per frame, bounce markers
- Summary panel — totals and per-type breakdown

---

## What I Learned

**Computer vision is harder than it looks.** Pretrained models work great on the data they were trained on, but the moment you move to a different domain (padel instead of general sports), performance drops hard. The gap between "works in the demo" and "works on real footage" is huge.

**Hybrid approaches work.** Combining deep learning (YOLO) with classical CV (HSV, background subtraction) gave better results than either alone. Sometimes the simplest solution (color thresholding) is the most reliable.

**Rules are underrated.** Everyone wants to train a neural network, but rule-based logic is fast, interpretable, and doesn't need labeled data. For a prototype, rules got me 80% of the way there.

**Tracking is critical.** Detection alone isn't enough — you need temporal consistency. EMA smoothing and missing-frame handling made the difference between a jittery mess and a usable trajectory.

**Evaluation is hard without ground truth.** I don't have labeled data to measure precision/recall, so I'm evaluating by eye. The system "feels" like it works, but I can't quantify how well.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Results saved to `results/`.

---

## Honest Limitations

- Ball detection still fails 30-40% of the time
- Shot classification rules are tuned to one video
- Racket detection is noisy and unreliable
- No rally segmentation or point tracking
- Assumes right-handed players
- No court boundary detection
- Can't handle fast camera pans or zooms

This is a prototype, not a production system. It demonstrates the concepts and shows what's possible with pretrained models and classical CV, but it would need significant work (fine-tuning, better tracking, learned classifiers) to be reliable on arbitrary padel footage.
