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

        print("Pipeline initialized")

    def run(self):
        print("Running padel analytics pipeline...")

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print("Error opening video")
            return

        frame_id = 0
        ball_history = []

        # get video FPS for accurate timestamps
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        print(f"Video FPS: {fps}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # hybrid detection: YOLO + OpenCV fallback for ball
            results, ball_pos, ball_method = self.detector.detect_with_ball_fallback(frame)

            # racket detection from same YOLO result (class 38, conf filtered)
            rackets = self.detector.detect_rackets(results[0])

            # ball tracking (pass pre-computed ball position)
            ball_history = self.tracker.update(frame_id, ball_pos)

            # racket tracking
            self.tracker.update_rackets(frame_id, rackets)

            # analytics
            self.analytics.process(frame_id, results)

            # shot classification
            shot = self.shot_classifier.update(frame_id, results, ball_history, fps)

            if shot:
                print(f"Shot detected: {shot} (ball via {ball_method})")

            # visualization
            frame = self.visualizer.draw(frame, results[0], shot, ball_pos, ball_method, rackets)

            cv2.imshow("Padel Analytics", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # save results
        shots = self.shot_classifier.get_shots()
        bounces = self.tracker.get_bounces()
        self.analytics.save_results(self.output_dir, shots, bounces)

        if ball_history:
            with open(os.path.join(self.output_dir, "ball_trajectory.json"), "w") as f:
                json.dump(ball_history, f, indent=4)

        racket_history = self.tracker.get_racket_history()
        if racket_history:
            with open(os.path.join(self.output_dir, "racket_tracking.json"), "w") as f:
                json.dump(racket_history, f, indent=4)

        bounces = self.tracker.get_bounces()
        if bounces:
            with open(os.path.join(self.output_dir, "bounces.json"), "w") as f:
                json.dump(bounces, f, indent=4)
            print(f"Bounces detected: {len(bounces)}")

        print("Pipeline completed")
