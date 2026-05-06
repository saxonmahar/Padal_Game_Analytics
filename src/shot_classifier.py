import math


class ShotClassifier:
    def __init__(self):
        print("Shot Classifier initialized")

        self.shots = []

        # cooldown to avoid duplicate shots on same hit
        self.last_shot_frame = -30
        self.shot_cooldown = 20

    def update(self, frame_id, results, ball_history, bounces, fps=30):
        """
        Classifies shots using ball trajectory + player position + racket proximity + bounce timing.
        """

        result = results[0]
        boxes = result.boxes

        players = 0
        player_boxes = []
        player_ids = []
        racket_boxes = []

        if boxes is not None:
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])

                if cls == 0:
                    players += 1
                    player_boxes.append(box.xyxy[0])

                    track_id = None
                    if hasattr(boxes, "id") and boxes.id is not None:
                        track_id = int(boxes.id[i])
                    player_ids.append(track_id)

                elif cls == 38:
                    racket_boxes.append(box.xyxy[0])

        ball_position = None
        if ball_history:
            ball_position = ball_history[-1]["position"]

        shot = self._classify(ball_history, players, player_boxes, racket_boxes, bounces, frame_id)

        if shot:
            player_id = self._nearest_player_id(ball_position, player_boxes, player_ids)
            timestamp = round(frame_id / fps, 2)

            self.shots.append({
                "frame_id": frame_id,
                "timestamp_seconds": timestamp,
                "shot_type": shot,
                "player_id": player_id
            })

        return shot

    def _classify(self, ball_history, players, player_boxes, racket_boxes, bounces, frame_id):

        if len(ball_history) < 6:
            return None

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

        ball_pos = positions[-1]
        shot = None

        # ---------------------------------------------------
        # 1. SMASH
        # Fast downward motion + ball is close to a player or racket
        # The proximity check avoids classifying random downward
        # ball movement (e.g. after a bounce) as a smash
        # ---------------------------------------------------
        if avg_dy > 18 and speed > 20:
            if self._ball_near_player_or_racket(ball_pos, player_boxes, racket_boxes, threshold=150):
                shot = "smash"

        # ---------------------------------------------------
        # 2. SERVE
        # Upward motion + high speed + a bounce happened recently
        # A serve in padel always follows a bounce (player drops
        # the ball, it bounces, then they hit it upward)
        # ---------------------------------------------------
        elif avg_dy < -10 and speed > 15:
            if self._recent_bounce(frame_id, bounces, within_frames=30):
                shot = "serve"

        # ---------------------------------------------------
        # 3. FOREHAND / BACKHAND
        # Horizontal motion + ball close to a player or racket
        # Proximity check ensures we're classifying an actual
        # hit, not just the ball drifting sideways
        # ---------------------------------------------------
        elif abs(avg_dx) > 5 and speed > 4:
            if self._ball_near_player_or_racket(ball_pos, player_boxes, racket_boxes, threshold=200):
                shot = self._classify_forehand_backhand(ball_pos, player_boxes, avg_dx)

        # ---------------------------------------------------
        # 4. RALLY (default)
        # Moderate speed, 2+ players visible, ball near someone
        # ---------------------------------------------------
        elif speed > 4 and players >= 2:
            if self._ball_near_player_or_racket(ball_pos, player_boxes, racket_boxes, threshold=250):
                shot = "rally"

        if shot:
            self.last_shot_frame = frame_id

        return shot

    def _ball_near_player_or_racket(self, ball_pos, player_boxes, racket_boxes, threshold):
        """
        Returns True if the ball is within `threshold` pixels of any player
        or racket bounding box center. This adds spatial context to shot
        detection — we only classify a shot if someone is actually near the ball.
        """

        if ball_pos is None:
            return False

        bx, by = ball_pos

        for box in player_boxes + racket_boxes:
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            dist = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
            if dist < threshold:
                return True

        return False

    def _recent_bounce(self, frame_id, bounces, within_frames=30):
        """
        Returns True if a bounce was detected within the last `within_frames` frames.
        Used to validate serve detection — a serve always follows a bounce.
        """

        for bounce in bounces:
            if 0 < frame_id - bounce["frame"] <= within_frames:
                return True

        return False

    def _classify_forehand_backhand(self, ball_pos, player_boxes, avg_dx):
        """
        Compare ball x position to nearest player center.
        Ball to the right of player = forehand (right-handed assumption).
        Ball to the left = backhand.
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

        return "forehand" if ball_x >= nearest_center_x else "backhand"

    def _nearest_player_id(self, ball_position, player_boxes, player_ids):
        """
        Find the track ID of the player closest to the ball.
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
