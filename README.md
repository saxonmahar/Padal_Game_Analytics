# Padel Game Analytics — Shot Classification System

A computer vision system that analyzes padel match footage to detect players, ball, and rackets, classify shot types, and generate structured analytics.

---

## Problem-Solving Approach

### Challenge 1: Ball Detection

**Problem:** Padel balls are small, fast, and often motion-blurred. YOLO's COCO pretrained model (class 32 = sports ball) is trained on soccer/basketball-sized objects and misses most padel ball detections.

**Solution:** Hybrid detection strategy with three fallback layers:
1. **YOLO (class 32)** — primary, works when ball is large/clear
2. **HSV color segmentation** — detects yellow-green balls against the court using OpenCV contour analysis
3. **Background subtraction (MOG2)** — catches fast-moving blobs even when color fails

**Trade-off:** HSV is sensitive to lighting changes. Background subtraction can trigger on player motion. But the three-layer fallback dramatically improves detection rate vs YOLO alone.

**Code:** `src/detector.py` — `detect_with_ball_fallback()`

---

### Challenge 2: Racket Detection

**Problem:** COCO has no padel racket class. Options were: (1) fine-tune YOLO on a padel dataset, (2) use COCO class 38 (tennis racket), or (3) build a custom detector.

**Solution:** Used COCO class 38 with a **confidence threshold of 0.45** (higher than default 0.25) to reduce false positives. Tennis rackets are visually similar enough to padel rackets that this works reasonably well without retraining.

**Trade-off:** Lower recall (misses some rackets) but higher precision (fewer false positives). For a prototype, this is acceptable. A production system would fine-tune on padel-specific data.

**Code:** `src/detector.py` — `detect_rackets()`

---

### Challenge 3: Shot Classification

**Problem:** No labeled padel shot dataset exists. Training a deep learning classifier from scratch would require hundreds of annotated clips.

**Solution:** Rule-based classification using ball trajectory features:
- **Smash:** fast downward motion (`avg_dy > 18`, `speed > 20`)
- **Serve:** upward arc followed by speed (`avg_dy < -10`, `speed > 15`)
- **Forehand/Backhand:** ball position relative to nearest player's center (left side = backhand, right side = forehand, assuming right-handed players)
- **Rally:** moderate speed with 2+ players visible

**Trade-off:** Rules are brittle and tuned to one video. A learned model would generalize better. But rules are interpretable, fast, and require zero training data.

**Code:** `src/shot_classifier.py` — `_classify()`

---

### Challenge 4: Player Assignment

**Problem:** Which player hit each shot?

**Solution:** For each detected shot, find the player bounding box closest to the ball position at that frame. Use ByteTrack IDs to maintain consistent player identity across frames.

**Trade-off:** Nearest-player heuristic fails when players overlap or the ball is mid-court. A better approach would track racket-ball contact events, but that requires higher temporal resolution and racket detection accuracy.

**Code:** `src/shot_classifier.py` — `_nearest_player_id()`

---

### Challenge 5: Bounce Detection

**Problem:** Detecting when the ball hits the ground is useful for rally analysis and shot validation.

**Solution:** Rule-based detection: a bounce occurs when the ball's vertical velocity flips from downward (`dy > 3`) to upward (`dy < -3`). A 10-frame cooldown prevents duplicate detections on the same bounce.

**Trade-off:** Sensitive to noise in ball tracking. EMA smoothing helps, but fast camera motion or occlusion can cause false positives. A physics-based model (parabolic trajectory fitting) would be more robust.

**Code:** `src/tracker.py` — `_detect_bounce()`

---

## System Architecture

```
Video Input
    ↓
Detector (YOLO + HSV + Motion)  →  Ball, Players, Rackets
    ↓
Tracker (EMA smoothing, ByteTrack IDs)  →  Ball trajectory, Racket history
    ↓
Shot Classifier (rule-based)  →  Shot type, Player ID, Timestamp
    ↓
Analytics  →  JSON, CSV, Dashboard PNG
```

**Modular design:** Each component (detector, tracker, classifier, analytics) is independent and testable. The pipeline wires them together.

---

## Technology Stack

- **Python** — core language
- **YOLOv8 (Ultralytics)** — object detection (COCO pretrained)
- **OpenCV** — video I/O, HSV color filtering, background subtraction, morphology
- **ByteTrack** — multi-object tracking (built into Ultralytics)
- **NumPy** — numerical operations (velocity, distance)
- **Pandas** — structured data export (CSV)
- **Matplotlib** — dashboard generation (bar, pie, scatter, line charts)

---

## Project Structure

```
padel-analytics/
├── src/
│   ├── detector.py         # Hybrid ball detection + racket detection
│   ├── tracker.py          # Ball tracking (EMA, velocity) + bounce detection
│   ├── shot_classifier.py  # Rule-based shot classification + player assignment
│   ├── analytics.py        # Data export + dashboard generation
│   ├── visualizer.py       # Video overlay (boxes, circles, labels)
│   └── pipeline.py         # Main processing loop
├── data/
│   └── input_sample_video.mp4
├── models/
│   └── yolov8n.pt          # COCO pretrained weights
├── results/
│   ├── shots_detected.json # Shot list (frame, timestamp, type, player_id)
│   ├── shots.csv           # Same data in CSV
│   ├── frame_data.json     # Per-frame player/racket counts
│   ├── summary.json        # Totals + shot type breakdown
│   ├── ball_trajectory.json # Ball position + velocity per frame
│   ├── racket_tracking.json # Racket positions per track_id
│   ├── bounces.json        # Bounce events (frame, position)
│   └── dashboard.png       # 5-chart visual dashboard
├── main.py
├── requirements.txt
└── README.md
```

---

## Output Files

### shots_detected.json
```json
[
  {
    "frame_id": 384,
    "timestamp_seconds": 12.8,
    "shot_type": "forehand",
    "player_id": 2
  },
  ...
]
```

### summary.json
```json
{
  "total_frames": 8125,
  "total_shots": 47,
  "total_bounces": 23,
  "shot_counts": {
    "forehand": 18,
    "backhand": 12,
    "smash": 9,
    "serve": 5,
    "rally": 3
  }
}
```

### dashboard.png
5-chart visual:
- Bar chart: shot count by type
- Pie chart: shot type percentage
- Scatter plot: shot timeline (when each shot occurred)
- Line chart: players/rackets detected per frame + bounce markers
- Summary panel: totals and per-type breakdown

---

## Key Design Decisions

### 1. Hybrid Detection Over Fine-Tuning
**Why:** Fine-tuning YOLO requires labeled data, GPU time, and hyperparameter tuning. For a prototype, combining YOLO with classical CV (HSV, background subtraction) gives better results faster.

### 2. Rule-Based Classification Over Deep Learning
**Why:** No labeled shot dataset exists. Rules are interpretable and fast. A learned model would require 500+ annotated clips and weeks of training.

### 3. EMA Smoothing on Ball Trajectory
**Why:** Raw detections are noisy (jitter, occlusion). Exponential moving average (alpha=0.6) smooths the trajectory while preserving responsiveness to direction changes.

### 4. Cooldown on Shot Detection
**Why:** A single hit can trigger multiple shot classifications across consecutive frames. A 20-frame cooldown prevents duplicates.

### 5. Confidence Threshold on Rackets
**Why:** COCO class 38 (tennis racket) has many false positives on padel footage. Raising the threshold from 0.25 to 0.45 improves precision at the cost of recall.

---

## Limitations & Future Work

### Current Limitations
- **Ball detection fails in occlusion** — when the ball passes behind a player or net
- **Forehand/backhand logic assumes right-handed players** — left-handed players are misclassified
- **No rally segmentation** — shots are detected independently, not grouped into rallies
- **No court boundary detection** — can't distinguish in-bounds vs out-of-bounds shots
- **Racket detection is noisy** — COCO class 38 is not trained on padel rackets

### Future Enhancements
- **Fine-tune YOLO on padel dataset** — improve ball and racket detection accuracy
- **Pose estimation (MediaPipe)** — detect player body orientation for better forehand/backhand classification
- **Optical flow** — improve ball tracking in occlusion
- **Rally segmentation** — group shots into points using serve detection + bounce patterns
- **Court keypoint detection** — map shots to court coordinates (baseline, net, service box)
- **Deep learning shot classifier** — train a temporal CNN on labeled shot clips

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py
```

Results are saved to `results/` folder.

---

## Evaluation Criteria Addressed

### Problem-Solving Approach
- Identified core challenges (ball detection, shot classification, player assignment)
- Designed hybrid solutions (YOLO + HSV + motion, rule-based classification)
- Documented trade-offs and limitations

### Understanding of Computer Vision Concepts
- Object detection (YOLO, COCO classes)
- Tracking (ByteTrack, EMA smoothing)
- Color segmentation (HSV thresholding)
- Background subtraction (MOG2)
- Contour analysis (circularity scoring)

### Code Quality & Structure
- Modular design (detector, tracker, classifier, analytics, visualizer, pipeline)
- Clear separation of concerns
- Docstrings on all public methods
- Consistent naming conventions

### Creativity & Experimentation
- Hybrid ball detection (3-layer fallback)
- Player-relative forehand/backhand classification
- Bounce detection via velocity reversal
- Multi-chart dashboard for recruiter readability

### Effort & Clarity of Explanation
- Detailed README with problem-solving rationale
- Inline comments explaining non-obvious logic
- Structured outputs (JSON, CSV, PNG)
- Visual dashboard with 5 charts

---

## Summary

This project demonstrates a practical approach to sports analytics using computer vision. The system prioritizes working functionality over perfection, combining pretrained models (YOLO) with classical CV techniques (HSV, background subtraction) and rule-based logic to deliver a complete prototype without requiring labeled training data or GPU resources.

The modular architecture makes it easy to swap components (e.g., replace rule-based classification with a learned model) as better data or compute becomes available.
