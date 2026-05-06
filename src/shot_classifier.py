import math


class ShotClassifier:
    def __init__(self):
        print("Shot Classifier initialized")

        self.history = []
        self.shots = []

        # cooldown to avoid duplicate shots on same hit
        self.last_shot_frame = -30
        self.shot_cooldown = 20

    def update(self, frame_id, results, ball_history):
        """
        Assigns shot type based on ball trajectory + players.
        Ball position is taken from ball_history (hybrid detector), not YOLO boxes.
        """

        result = results[0]
        boxes = result.boxes

        players = 0
        player_boxes = []

        # count players and store their bounding boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    players += 1
                    player_boxes.append(box.xyxy[0])

        # latest ball position from tracker history
        ball_position = None
        if ball_history:
            ball_position = ball_history[-1]["position"]

        self.history.append({
            "frame": frame_id,
            "players": players,
            "ball": ball_position
        })

        shot = self._classify(ball_history, players, player_boxes, frame_id)

        if shot:
            self.shots.append({
                "frame_id": frame_id,
                "shot_type": shot
            })

        return shot

    def _classify(self, ball_history, players, player_boxes, frame_id):

        if len(ball_history) < 6:
            return None

        # cooldown check — avoid duplicate detections on same hit
        if frame_id - self.last_shot_frame < self.shot_cooldown:
            return None

        window = ball_history[-5:]
        positions = [p["position"] for p in window if p["position"]]

        if len(positions) < 5:
            return None

        dx_list = []
        dy_list = []

        for i in range(1, len(positions)):
            dx_list.append(positions[i][0] - positions[i - 1][0])
            dy_list.append(positions[i][1] - positions[i - 1][1])

        avg_dx = sum(dx_list) / len(dx_list)
        avg_dy = sum(dy_list) / len(dy_list)

        speed = sum(
            math.sqrt(dx * dx + dy * dy)
            for dx, dy in zip(dx_list, dy_list)
        ) / len(dx_list)

        shot = None

        # ---------------------------------------------------
        # 1. SMASH (fast downward motion)
        # ---------------------------------------------------
        if avg_dy > 18 and speed > 20:
            shot = "smash"

        # ---------------------------------------------------
        # 2. SERVE (strong upward then downward arc, high speed)
        # ---------------------------------------------------
        elif avg_dy < -10 and speed > 15:
            shot = "serve"

        # ---------------------------------------------------
        # 3. FOREHAND / BACKHAND
        # Uses ball position relative to nearest player center
        # to determine which side of the body the hit came from
        # ---------------------------------------------------
        elif abs(avg_dx) > 5 and speed > 4:
            ball_pos = positions[-1]
            shot = self._classify_forehand_backhand(ball_pos, player_boxes, avg_dx)

        # ---------------------------------------------------
        # 4. DEFAULT RALLY
        # ---------------------------------------------------
        elif speed > 4 and players >= 2:
            shot = "rally"

        if shot:
            self.last_shot_frame = frame_id

        return shot

    def _classify_forehand_backhand(self, ball_pos, player_boxes, avg_dx):
        """
        Determine forehand or backhand by comparing ball x position
        to the nearest player's center x.
        If ball is to the right of player center -> forehand (right-handed assumption)
        If ball is to the left of player center  -> backhand
        Falls back to direction-based if no players detected.
        """

        if not player_boxes:
            # fallback: use ball direction
            return "forehand" if avg_dx > 0 else "backhand"

        ball_x = ball_pos[0]

        # find nearest player by horizontal distance
        nearest_center_x = None
        min_dist = float("inf")

        for box in player_boxes:
            x1, y1, x2, y2 = box
            player_cx = float((x1 + x2) / 2)
            dist = abs(player_cx - ball_x)

            if dist < min_dist:
                min_dist = dist
                nearest_center_x = player_cx

        if nearest_center_x is None:
            return "forehand" if avg_dx > 0 else "backhand"

        # ball to the right of player = forehand side
        if ball_x >= nearest_center_x:
            return "forehand"
        else:
            return "backhand"

    def get_shots(self):
        return self.shots
