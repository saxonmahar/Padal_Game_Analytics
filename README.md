#  Padel Analytics System

A computer vision-based system for analyzing padel tennis matches using YOLOv8, object tracking, and custom analytics.

---

## Project Overview

This project processes padel match videos to:
- Detect players and ball using YOLOv8
- Track objects across frames
- Classify shot types (serve, volley, smash, etc.)
- Generate match statistics and analytics
- Output annotated videos and structured data

---

## 📁 Project Structure

padel-analytics/
│
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── shot_classifier.py
│   ├── analytics.py
│   ├── visualizer.py
│   └── pipeline.py
│
├── data/
│   └── input_video.mp4
│
├── models/
│   └── yolov8n.pt
│
├── results/
│   ├── shots.json
│   ├── shots.csv
│   ├── dashboard.png
│   └── output_annotated.mp4
│
├── main.py
├── demo_mode.py
├── requirements.txt
└── README.md

---

##  Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd padel-analytics


2. Create virtual environment (recommended)
python -m venv env
to activate the environment
source env/scripts/activate(in windows)
source env/bin/activate(in mac/linux)

3. Install dependencies
pip install -r requirements.txt

Run the Project
python main.py

---