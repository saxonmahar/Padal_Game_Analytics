# 🎾 Padel Game Analytics — Shot Classification System
# 🎾 Padel Game Analytics — Shot Classification System

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:1a1f2e,100:0D1117&height=180&section=header&text=Padel%20Analytics&fontColor=58a6ff&fontSize=42&fontAlignY=38&desc=Computer%20Vision%20%7C%20YOLOv8%20%7C%20Tracking%20%7C%20Shot%20Analysis&descAlignY=58&descSize=16&descColor=8b949e&animation=fadeIn" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=20&duration=3000&pause=800&color=58A6FF&center=true&vCenter=true&width=700&lines=YOLOv8+Object+Detection+%2B+ByteTrack;Ball+Trajectory+%26+Shot+Classification;Real-Time+Padel+Match+Analytics;Computer+Vision+End-to-End+Pipeline" />
</p>

---

## 🚀 Project Overview

This project is a **Computer Vision-based Padel Match Analytics System** that:

- 🎯 Detects players & ball using YOLOv8  
- 🏃 Tracks ball trajectory across frames  
- 🎾 Classifies shots (smash, lob, rally)  
- 📊 Generates analytics (CSV + JSON + dashboard)  
- 🎥 Outputs annotated match video  

---

## ⚙️ Tech Stack

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv8-111111?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge"/>

</p>

---

## 📁 Project Structure

```bash
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
│   ├── summary.json
│   ├── dashboard.png
│   ├── ball_trajectory.json
│   └── output_annotated.mp4
│
├── main.py
├── demo_mode.py
├── requirements.txt
└── README.md


## 🔥 Pipeline Workflow

- 🎥 Load padel match video  
- 🧠 YOLOv8 detects players & ball  
- 🏃 ByteTrack tracks objects  
- 📍 Ball trajectory computed  
- 🎾 Shot classification (rule-based)  
- 📊 Analytics stored (frame-wise stats)  
- 📈 Dashboard generated (matplotlib)  
- 🎬 Annotated video exported  

---

## 📊 Output Files

- `shots.json` → shot events  
- `shots.csv` → frame-level analytics  
- `summary.json` → overall stats  
- `ball_trajectory.json` → ball movement path  
- `dashboard.png` → visualization graphs  
- `output_annotated.mp4` → final processed video  

---

## ✨ Features Implemented

- ✔ Player detection  
- ✔ Ball detection  
- ✔ Ball tracking (trajectory)  
- ✔ Shot classification (smash / lob / rally)  
- ✔ Real-time visualization  
- ✔ Analytics dashboard  
- ✔ Structured data export  

---

## 📈 Dashboard (Matplotlib)

The system generates:

- 📊 Players detected per frame graph  
- 🎾 Shot distribution (smash / lob / rally)  
- 📈 Total shots overview  

---

## 🚀 Future Improvements

- 🎯 Forehand / Backhand classification (pose estimation)  
- 🧠 Deep learning-based shot classifier  
- 🎾 Racket detection model integration  
- 📡 Real-time live match analysis  
- 🏆 Rally segmentation (point start/end detection)  

---

## 🧠 Key Idea

A real-world sports analytics system inspired by professional coaching tools, built using YOLOv8 + tracking + rule-based AI.

---

