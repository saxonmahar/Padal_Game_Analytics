import math


class ShotClassifier:
    def __init__(self):
        print("🎾 Shot Classifier initialized")

        self.history = []
        self.shots = []

    def update(self, frame_id, results):
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

                # sports ball
                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    ball_position = (cx, cy)

        # store frame info
        frame_data = {
            "frame_id": frame_id,
            "players": players,
            "ball_position": ball_position
        }

        self.history.append(frame_data)

        # classify shot
        shot = self._classify()

        if shot:
            self.shots.append({
                "frame_id": frame_id,
                "shot_type": shot
            })

        return shot

    def _classify(self):
        if len(self.history) < 3:
            return None

        last = self.history[-1]
        prev = self.history[-2]

        if not last["ball_position"] or not prev["ball_position"]:
            return None

        x1, y1 = prev["ball_position"]
        x2, y2 = last["ball_position"]

        dx = x2 - x1
        dy = y2 - y1

        speed = math.sqrt(dx**2 + dy**2)

        # 🔥 RULES (tune these thresholds)

        # 1. Smash → fast downward motion
        if dy > 15 and speed > 20:
            return "smash"

        # 2. Lob → upward motion
        if dy < -10:
            return "lob"

        # 3. Rally → moderate movement + players active
        if speed > 5 and last["players"] >= 2:
            return "rally"

        return None

    def get_shots(self):
        return self.shots