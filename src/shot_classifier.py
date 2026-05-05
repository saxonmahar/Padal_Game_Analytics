import math


class ShotClassifier:
    def __init__(self):
        print("🎾 Shot Classifier initialized")

        self.history = []
        self.shots = []

    def update(self, frame_id, results, ball_history):
        """
        Uses ball movement + speed for classification
        """

        result = results[0]
        boxes = result.boxes

        ball_position = None
        players = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    players += 1

                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    ball_position = (cx, cy)

        self.history.append({
            "frame": frame_id,
            "players": players,
            "ball": ball_position
        })

        shot = self._classify(ball_history, players)

        if shot:
            self.shots.append({
                "frame_id": frame_id,
                "shot_type": shot
            })

        return shot

    def _classify(self, ball_history, players):
        """
        REAL upgrade: speed + direction logic
        """

        if len(ball_history) < 3:
            return None

        p1 = ball_history[-2]["position"]
        p2 = ball_history[-1]["position"]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        speed = math.sqrt(dx * dx + dy * dy)

        # 🔥 SMASH → fast downward
        if dy > 20 and speed > 25:
            return "smash"

        # 🔥 LOB → upward motion
        if dy < -15:
            return "lob"

        # 🔥 RALLY → normal movement
        if speed > 5 and players >= 2:
            return "rally"

        return None

    def get_shots(self):
        return self.shots