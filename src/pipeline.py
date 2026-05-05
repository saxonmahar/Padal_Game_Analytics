import cv2
import os
from src.detector import Detector
from src.tracker import Tracker
from src.analytics import Analytics


class Pipeline:
    def __init__(self, video_path, model_path, output_dir):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = Detector(self.model_path)
        self.tracker = Tracker()
        self.analytics = Analytics()

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

            # 🔥 TRACKING (YOLO + ByteTrack)
            results = self.detector.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            result = results[0]

            # 📊 ANALYTICS
            self.analytics.process(frame_count, results)

            # 🎨 VISUALIZATION
            annotated_frame = result.plot()

            out.write(annotated_frame)

            print(f"Processing frame {frame_count}")

        cap.release()
        out.release()

        # 📊 SAVE ANALYTICS AFTER LOOP
        self.analytics.save_results(self.output_dir)

        print("📊 Generating analytics...")
        print("💾 Saving results...")

        print("✅ Pipeline finished successfully")
        print(f"📁 Output saved at: {output_path}")