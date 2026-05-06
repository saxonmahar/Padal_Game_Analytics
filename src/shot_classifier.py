import math


class ShotClassifier:
    def __init__(self):
        print("🎾 Shot Classifier initialized (fixed)")

        self.history = []
        self.shots = []

        # 🔥 prevents duplicate detections
        self.last_shot = None
        self.last_shot_frame = -1
        self.cooldown = 8  # frames

    def update(self, frame_id, results, ball_history):
        """
        Uses ball movement + speed + bounce detection
        with anti-duplicate event system
        """

        result = results[0]
        boxes = result.boxes

        ball_position = None
        players = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                # person
                if cls == 0:
                    players += 1

                # ball
                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    ball_position = (cx, cy)

        # store frame info
        self.history.append({
            "frame": frame_id,
            "players": players,
            "ball": ball_position
        })

        shot = self._classify(ball_history, players)

        # -------------------------------
        # 🔥 ANTI-DUPLICATE FILTER
        # -------------------------------
        if shot:
            if (
                shot != self.last_shot or
                frame_id - self.last_shot_frame > self.cooldown
            ):
                self.shots.append({
                    "frame_id": frame_id,
                    "shot_type": shot
                })

                self.last_shot = shot
                self.last_shot_frame = frame_id

                return shot

        return None

    def _classify(self, ball_history, players):
        """
        Improved classification:
        - Bounce detection
        - Speed + direction logic
        """

        if not ball_history or len(ball_history) < 3:
            return None

        # ensure valid positions
        if not all(ball_history[-i]["position"] for i in [1, 2, 3]):
            return None

        p0 = ball_history[-3]["position"]
        p1 = ball_history[-2]["position"]
        p2 = ball_history[-1]["position"]

        # movement
        dy1 = p1[1] - p0[1]
        dy2 = p2[1] - p1[1]

        dx = p2[0] - p1[0]
        dy = dy2

        speed = math.sqrt(dx * dx + dy * dy)

        # -------------------------------
        # 🔥 BOUNCE DETECTION
        # down → up transition
        # -------------------------------
        if dy1 > 6 and dy2 < -6 and speed > 6:
            return "bounce"

        # -------------------------------
        # 🔥 SMASH (fast downward)
        # -------------------------------
        if dy > 20 and speed > 25:
            return "smash"

        # -------------------------------
        # 🔥 LOB (upward motion)
        # -------------------------------
        if dy < -15:
            return "lob"

        # -------------------------------
        # 🔥 RALLY (normal play)
        # -------------------------------
        if speed > 5 and players >= 2:
            return "rally"

        return None

    def get_shots(self):
        return self.shots