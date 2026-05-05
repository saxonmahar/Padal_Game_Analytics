import cv2
import os
from src.detector import Detector


class Pipeline:
    def __init__(self, video_path, model_path, output_dir):
        self.video_path = video_path
        self.model_path = model_path
        self.output_dir = output_dir

        # Create output folder if not exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize detector
        self.detector = Detector(self.model_path)

        print("📦 Pipeline initialized")
        print(f"Video: {self.video_path}")
        print(f"Model: {self.model_path}")
        print(f"Output: {self.output_dir}")

    def run(self):
        print("🎬 Running padel analytics pipeline...")

        # STEP 1: Load video (FIXED - use self.video_path)
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("❌ Error: Cannot open video")
            return

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video path
        output_path = os.path.join(self.output_dir, "output_annotated.mp4")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        # STEP 2: Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # STEP 3: Detection
            results = self.detector.detect(frame)

            # Draw detections on frame (IMPORTANT FIX)
            annotated_frame = results[0].plot()

            # Save frame to output video
            out.write(annotated_frame)

            print(f"Processing frame {frame_count}")

        # Cleanup
        cap.release()
        out.release()

        # STEP 4: Analytics placeholder
        print("Generating analytics...")

        # STEP 5: Save results placeholder
        print("Saving results...")

        print("✅ Pipeline finished successfully")
        print(f"📁 Output saved at: {output_path}")