import math


class ShotClassifier:
    def __init__(self):
        print("Shot Classifier initialized")

        self.history = []
        self.shots = []

        # cooldown to avoid duplicate shots on same hit
        self.last_shot_frame = -30
        self.shot_cooldown = 20

    def update(self, frame_id, results, ball_history, fps=30):
        """
        Assigns shot type based on ball trajectory + players.
        Ball position is taken from ball_history (hybrid detector), not YOLO boxes.
        fps is used to compute timestamp in seconds.
        """

        result = results[0]
        boxes = result.boxes

        players = 0
        player_boxes = []
        player_ids = []

        # count players and store their bounding boxes + track IDs
        if boxes is not None:
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])

                if cls == 0:
                    players += 1
                    player_boxes.append(box.xyxy[0])

                    # get track ID if available
                    track_id = None
                    if hasattr(boxes, "id") and boxes.id is not None:
                        track_id = int(boxes.id[i])
                    player_ids.append(track_id)

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
            # find nearest player ID to the ball at this frame
            player_id = self._nearest_player_id(ball_position, player_boxes, player_ids)

            timestamp = round(frame_id / fps, 2)

            self.shots.append({
                "frame_id": frame_id,
                "timestamp_seconds": timestamp,
                "shot_type": shot,
                "player_id": player_id
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
        # 2. SERVE (strong upward arc, high speed)
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
            return "forehand" if avg_dx > 0 else "backhand"

        ball_x = ball_pos[0]

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

        if ball_x >= nearest_center_x:
            return "forehand"
        else:
            return "backhand"

    def _nearest_player_id(self, ball_position, player_boxes, player_ids):
        """
        Find the track ID of the player closest to the ball.
        Returns track_id or None if no players detected.
        """

        if not player_boxes or ball_position is None:
            return None

        ball_x, ball_y = ball_position
        min_dist = float("inf")
        nearest_id = None

        for box, pid in zip(player_boxes, player_ids):
            x1, y1, x2, y2 = box
            player_cx = float((x1 + x2) / 2)
            player_cy = float((y1 + y2) / 2)

            dist = math.sqrt((player_cx - ball_x) ** 2 + (player_cy - ball_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                nearest_id = pid

        return nearest_id

    def get_shots(self):
        return self.shots
