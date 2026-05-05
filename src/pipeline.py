import cv2
import os
from src.detector import Detector
from src.tracker import Tracker
from src.analytics import Analytics
from src.shot_classifier import ShotClassifier


class Pipeline:
    def __init__(self, video_path, model_path, output_dir):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = Detector(self.model_path)
        self.tracker = Tracker()
        self.analytics = Analytics()
        self.shot_classifier = ShotClassifier()  # ✅ FIXED

        print("📦 Pipeline initialized")
        print(f"Video: {self.video_path}")
        print(f"Model: {self.model_path}")
        print(f"Output: {self.output_dir}")

    def run(self):
        print("🎬 Running padel analytics pipeline...")

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("❌ Error: Cannot open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(self.output_dir, "output_annotated.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 🔥 TRACKING
            results = self.detector.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # 📊 ANALYTICS (per frame)
            self.analytics.process(frame_count, results)

            # 🎾 SHOT CLASSIFICATION (per frame) ✅ FIXED
            shot = self.shot_classifier.update(frame_count, results)

            if shot:
                print(f"🎾 Shot detected at frame {frame_count}: {shot}")

            # 🎨 VISUALIZATION
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

            print(f"Processing frame {frame_count}")

        cap.release()
        out.release()

        # 📊 SAVE RESULTS
        self.analytics.save_results(self.output_dir)

        # 🎾 SAVE SHOTS
        shots = self.shot_classifier.get_shots()
        import json
        with open(os.path.join(self.output_dir, "shots_detected.json"), "w") as f:
            json.dump(shots, f, indent=4)

        print("📊 Generating analytics...")
        print("💾 Saving results...")

        print("✅ Pipeline finished successfully")
        print(f"📁 Output saved at: {output_path}")