from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        print("📦 Loading YOLO model...")
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs YOLO detection on a single frame
        """
        results = self.model(frame)
        return results