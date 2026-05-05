# Padel Game Analytics — Shot Classification System

## Overview

This project is an end-to-end computer vision system for analyzing padel tennis match footage. It detects and tracks players and the ball, performs basic shot classification, and generates structured match analytics and visual insights.

The system is built using YOLOv8 for object detection, ByteTrack for tracking, and rule-based logic for initial shot classification.

---

## Objectives

- Detect players and ball from match video
- Track ball movement across frames
- Classify basic shot types (smash, lob, rally)
- Generate structured analytical outputs
- Produce annotated video with overlays
- Generate statistical dashboards for insights

---

## System Architecture

The pipeline follows a sequential processing workflow:

1. Video input is loaded into the pipeline
2. YOLOv8 detects players and ball
3. ByteTrack assigns consistent object identities across frames
4. Ball trajectory is computed from frame-level positions
5. Rule-based logic classifies shots (smash, lob, rally)
6. Frame-wise analytics are recorded in structured format
7. Matplotlib generates performance dashboards
8. Final annotated video is exported

---

## Technology Stack

- Python (core implementation)
- YOLOv8 (object detection)
- OpenCV (video processing)
- ByteTrack (multi-object tracking)
- Pandas (data processing and storage)
- NumPy (numerical computation)
- Matplotlib (data visualization)

---

## Project Structure


padel-analytics/
│
├── src/
│ ├── detector.py
│ ├── tracker.py
│ ├── shot_classifier.py
│ ├── analytics.py
│ ├── visualizer.py
│ └── pipeline.py
│
├── data/
│ └── input_video.mp4
│
├── models/
│ └── yolov8n.pt
│
├── results/
│ ├── shots.json
│ ├── shots.csv
│ ├── summary.json
│ ├── ball_trajectory.json
│ ├── dashboard.png
│ └── output_annotated.mp4
│
├── main.py
├── demo_mode.py
├── requirements.txt
└── README.md






---

## Output Artifacts

The system generates the following outputs:

- shots.json → shot-level events
- shots.csv → frame-wise analytics
- summary.json → overall match statistics
- ball_trajectory.json → ball movement path
- dashboard.png → visual analytics dashboard
- output_annotated.mp4 → processed video with overlays

---

## Features Implemented

- Player detection using YOLOv8
- Ball detection and tracking
- Ball trajectory reconstruction
- Rule-based shot classification (smash, lob, rally)
- Real-time video visualization
- Automated analytics generation
- Structured data export (JSON and CSV)
- Matplotlib-based dashboard generation

---

## Dashboard Analytics

The system provides visual insights including:

- Player count per frame over time
- Shot type distribution (smash, lob, rally)
- Overall shot frequency statistics

---

## Future Enhancements

- Forehand and backhand classification using pose estimation
- Deep learning-based shot classification model
- Racket detection module integration
- Real-time live match analysis
- Rally segmentation and point detection system

---

## Key Concept

This project demonstrates a practical sports analytics pipeline combining computer vision, object tracking, and rule-based reasoning to simulate real-world performance analysis systems used in sports technology.

---

## Summary

A modular and scalable computer vision system designed to analyze padel matches, extract meaningful gameplay insights, and generate structured performance analytics.