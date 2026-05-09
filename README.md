# Padel Game Analytics — Shot Classification System

I built this as part of a computer vision assignment. The task sounds straightforward — detect the ball, players, and rackets in a padel match video, classify shot types, output analytics. But the hard part is that padel balls are tiny (around 15 pixels wide), move fast enough to blur, and disappear behind players constantly. There's no pretrained model for padel, no labeled dataset, and the camera is mounted high overhead which breaks most standard assumptions about player orientation. So the real challenge wasn't "which model do I use" — it was figuring out how to build something meaningful with the tools available.

I want to be upfront: this is a prototype. Some parts work well, some are still rough, and I know exactly where the gaps are. I've tried to document all of it honestly, including what I started with, what I improved, and what I'd do next.

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
Detector   →  YOLO (players, rackets) + HSV + Motion fusion (ball)
    ↓
Tracker    →  Kalman filter · jump rejection · bounce detection
    ↓
Classifier →  Rules: player-relative height + direction + bounce timing + speed spike
    ↓
Analytics  →  JSON · CSV · summary · dashboard PNG
    ↓
Visualizer →  Ball trail · bounding boxes · shot labels on video
```

---

## Where I Started and What I Changed

### Ball Detection

**Where I started:** YOLO alone. It detected the ball in under 10% of frames. The model was trained on soccer and basketballs — a padel ball is 15 pixels wide and moves fast. It just didn't work.

**What I tried first:** Lowering the confidence threshold. Got more false positives, not more ball. Tried a bigger YOLO model — slightly better, much slower.

**What I built:** A three-layer cascade — YOLO first, then HSV color segmentation (padel balls are yellow-green), then background subtraction (anything moving against the static court). Detection rate went from ~10% to ~60–70%.

I chose HSV and background subtraction over retraining a model because the problem was a domain mismatch, not a model architecture problem. YOLO knows what a ball looks like — it just never saw a padel ball. Retraining would fix that, but it needs labeled data I don't have. Classical CV doesn't need labels — it just needs the right observation, and "yellow-green circular thing that's moving" is a pretty good description of a padel ball.

**Problem I hit:** HSV was picking up court markings and player clothing. Too many false positives were being treated as real balls, which broke the trajectory and the classifier downstream.

**What I improved:** Added HSV + motion fusion. Instead of accepting either method alone, I now require both to agree within 40 pixels. A real ball is yellow-green AND moving. A court marking is yellow-green but static. A player's arm moves but isn't yellow-green. Requiring both conditions kills most false positives.

Also added a max jump distance check in the tracker — if the detected position jumps more than 120 pixels from the last known position in one frame, it's rejected as a false detection before it even reaches the Kalman filter.

---

### Ball Tracking

**Where I started:** EMA (Exponential Moving Average) smoothing with a simple "hold last position for 8 frames" approach during occlusion.

**The problem:** EMA just blends old and new positions. It has no model of motion. During occlusion it holds the last position, which is wrong — the ball is still moving. And noisy detections from HSV false positives were throwing off the trajectory.

**What I implemented:** A 2D Kalman filter. The state vector is `[x, y, vx, vy]` — position and velocity. Each frame it predicts where the ball should be based on its current velocity, then blends that prediction with the actual detection. During occlusion it predicts using velocity instead of holding the last position. Noisy detections get weighted against the predicted trajectory, so false positives have less impact.

The key tuning parameter is measurement noise `R=8.0`. Lower R means trust the detector more (responsive but noisy). Higher R means trust the model more (smooth but laggy). 8.0 is a balanced middle ground for padel ball speed.

---

### Shot Classification

**Where I started:** Simple motion thresholds. `avg_dy > 18 = smash`, `avg_dy < -10 = serve`. It fired constantly on everything and called almost every shot a serve.

**The problem:** Raw pixel dy is camera-angle dependent. This video is shot from a high overhead angle. From that perspective, almost every shot toward the far end of the court has negative dy in pixel space — so everything looked like a serve.

**What I improved:**

- **Player-relative height** — instead of raw dy, I compute where the ball is relative to the nearest player's bounding box. Above the head = smash. Waist height = forehand/backhand. This removes camera angle bias completely.

- **Player-relative direction** — instead of raw dx, I check whether the ball is to the left or right of the nearest player's center. Camera-angle independent forehand/backhand.

- **Bounce timing for serve** — serve only fires if there were at least 2 bounces in the last 60 frames. A padel serve always follows the player bouncing the ball. Requiring 2 bounces (not 1) filters out background subtraction noise that was generating fake bounces constantly.

- **Speed spike detection** — instead of classifying every frame, I now only classify when ball speed increases by more than 8px/frame (a hit causes acceleration) or the ball is within 80px of a player. This fires once per hit event instead of across many frames.

| Shot | Conditions |
|---|---|
| Smash | Fast downward motion + ball within 150px of player |
| Serve | Upward motion + high speed + 2+ recent bounces |
| Forehand | Ball right of nearest player center + speed > 5 + within 150px |
| Backhand | Ball left of nearest player center + speed > 5 + within 150px |
| Rally | Moderate speed + 2+ players visible + within 250px |

---

### Evaluation

I built `evaluate.py` against 25 manually labeled shots. Small sample, but having a number is better than guessing.

```bash
python evaluate.py
```

The evaluation now outputs a confusion matrix alongside accuracy and per-class breakdown. The confusion matrix shows exactly which shot types are being confused with which — much more useful than just an overall accuracy number.

Full report: `results/evaluation_report.json`

---

### Why I Didn't Use MediaPipe Pose Estimation

I thought about this one quite a bit. Pose estimation is genuinely the right tool for forehand/backhand classification — wrist position, elbow angle, shoulder rotation. If you can see those, you don't need to guess which side of the body the ball is on.

But I didn't use it, and here's why.

First, speed. On my machine without a GPU, MediaPipe runs at around 15fps. The pipeline is already slower than real time. Stacking pose estimation on top would make it noticeably worse.

Second, the camera angle. The test video is shot from high up, looking down at the court. Pose estimation is trained mostly on people filmed from the front or side. From above, limbs overlap, shoulders are foreshortened, and the keypoints become unreliable. I wasn't confident it would actually help on this specific footage.

Third, timing. Adding a new dependency that might fail or slow things down right before submission felt like the wrong call.

What I did instead was make the existing features camera-angle independent — player-relative height and player-relative direction. Same problem, different approach, no new dependencies.

If I kept working on this, I'd add MediaPipe on every 3rd frame with a fallback to the current rules when pose detection fails.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| YOLOv8 (Ultralytics) | Object detection |
| ByteTrack | Multi-object tracking with persistent IDs |
| OpenCV | HSV filtering, background subtraction, video I/O |
| NumPy | Kalman filter matrices, velocity calculations |
| Pandas | CSV export |
| Matplotlib | Dashboard generation |

---

## Project Structure

```
src/
├── detector.py        # Hybrid ball detection (YOLO + HSV+motion fusion) + racket detection
├── tracker.py         # Kalman filter tracking + jump rejection + bounce detection
├── shot_classifier.py # Shot classification + speed spike detection + player assignment
├── analytics.py       # Data export + dashboard
├── visualizer.py      # Video overlay — ball trail, boxes, shot labels
└── pipeline.py        # Main processing loop

data/
└── ground_truth.json  # 25 manually labeled shots for evaluation

evaluate.py            # Accuracy + confusion matrix evaluation script
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
| `evaluation_report.json` | Accuracy + confusion matrix vs ground truth |
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

## Demo

[Watch the annotated output video on Google Drive](https://drive.google.com/file/d/1YlN1oxWtan19uahVHP84UZVZ0pwALs4x/view?usp=sharing)

---

## Dashboard

![Dashboard](assets/dashboard.png)

[View full dashboard image on Google Drive](https://drive.google.com/file/d/1Cor6Yn_4D_HueMuBuFO8ipAeSKvV4WSZ/view?usp=sharing)

---

## Real Limitations

Ball detection still misses frames during occlusion and fast motion. The Kalman filter predicts through short gaps but can't recover from long ones.

Shot classification rules are tuned to this video. A different camera angle, court, or lighting will probably need retuning. The thresholds are hand-picked, not learned.

Racket detection uses COCO class 38 (tennis racket) — noisy on padel footage. It contributes to proximity checks but I wouldn't rely on it alone.

Forehand/backhand assumes right-handed players. Left-handed players will be misclassified.

25 ground truth samples is not enough for a statistically meaningful evaluation. It tells me where to look, not how good the system actually is.

---

## What I'd Do With More Time

- Fine-tune YOLO on padel-specific data — the single biggest improvement available
- Add court homography to map pixel positions to real court coordinates — makes all spatial features truly camera-angle independent
- Add MediaPipe pose estimation on every 3rd frame with fallback to current rules
- Train a temporal classifier on labeled shot clips to replace the hand-tuned rules
- Label more ground truth samples — 25 is not enough
- Add rally segmentation to group shots into points
- Build the evaluation script first next time — it would have helped tune thresholds much faster
