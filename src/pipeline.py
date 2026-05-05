import cv2
import os
import json

from src.detector import Detector
from src.tracker import Tracker
from src.analytics import Analytics
from src.shot_classifier import ShotClassifier
from src.visualizer import Visualizer


class Pipeline:
    def __init__(self, video_path, model_path, output_dir):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = Detector(self.model_path)
        self.tracker = Tracker()
        self.analytics = Analytics()
        self.shot_classifier = ShotClassifier()
        self.visualizer = Visualizer()

        print("📦 Pipeline initialized")

    def run(self):
        print("🎬 Running padel analytics pipeline...")

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("❌ Error opening video")
            return

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # YOLO tracking
            results = self.detector.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # 🔵 BALL TRACKING
            ball_history = self.tracker.update(frame_id, results)

            # 📊 ANALYTICS
            self.analytics.process(frame_id, results)

            # 🎾 SHOT CLASSIFICATION (NOW USING BALL HISTORY)
            shot = self.shot_classifier.update(frame_id, results, ball_history)

            if shot:
                print(f"🎾 Shot detected: {shot}")

            # 🎨 VISUALIZATION
            frame = self.visualizer.draw(frame, results[0], shot)

            cv2.imshow("Padel Analytics", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # SAVE RESULTS
        self.analytics.save_results(self.output_dir, self.shot_classifier.get_shots())

        with open(os.path.join(self.output_dir, "ball_trajectory.json"), "w") as f:
            json.dump(ball_history, f, indent=4)

        print("✅ Pipeline completed")