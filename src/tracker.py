class Tracker:
    def __init__(self):
        print("📦 Tracker initialized (ByteTrack via YOLO)")

    def update(self, results):
        """
        Uses YOLO tracking output (ByteTrack internally via ultralytics)
        """
        # results already contains tracking info if used with .track()
        return results