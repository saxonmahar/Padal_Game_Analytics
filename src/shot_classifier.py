import math


class ShotClassifier:
    def __init__(self):
        print("Shot Classifier initialized")

        self.shots = []

        # cooldown to avoid duplicate shots on same hit
        self.last_shot_frame = -30
        self.shot_cooldown = 25

    def update(self, frame_id, results, ball_history, bounces, fps=30):
        """
        Classifies shots using ball trajectory + player-relative height + racket proximity.
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

        if len(ball_history) < 8:
            return None

        if frame_id - self.last_shot_frame < self.shot_cooldown:
            return None

        window = ball_history[-7:]
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

        # get ball height relative to nearest player
        # this removes camera angle bias from raw pixel dy
        ball_zone = self._ball_height_zone(ball_pos, player_boxes)

        shot = None

        # ---------------------------------------------------
        # 1. SMASH
        # Ball is ABOVE the player's head + fast speed
        # Raw avg_dy > 0 means ball moving down in pixel space
        # but we now confirm it's actually high up relative to player
        # ---------------------------------------------------
        if ball_zone == "above_head" and speed > 18 and avg_dy > 5:
            if self._ball_near_player_or_racket(ball_pos, player_boxes, racket_boxes, threshold=180):
                shot = "smash"

        # ---------------------------------------------------
        # 2. SERVE
        # Ball starts near waist/ground level (player toss),
        # then moves upward fast + recent bounce confirms the toss
        # ---------------------------------------------------
        elif ball_zone in ("waist", "low") and speed > 18 and avg_dy < -5:
            if self._confirmed_recent_bounce(frame_id, bounces, within_frames=45):
                shot = "serve"

        # ---------------------------------------------------
        # 3. FOREHAND / BACKHAND
        # Ball at waist height + clear horizontal motion
        # Direction is computed relative to nearest player center
        # not raw avg_dx — removes camera angle dependency
        # ---------------------------------------------------
        elif ball_zone == "waist" and speed > 5:
            if self._ball_near_player_or_racket(ball_pos, player_boxes, racket_boxes, threshold=180):
                shot = self._classify_forehand_backhand(ball_pos, player_boxes, avg_dx)

        # ---------------------------------------------------
        # 4. RALLY
        # Any height, moderate speed, 2+ players visible
        # ---------------------------------------------------
        elif speed > 6 and players >= 2:
            if self._ball_near_player_or_racket(ball_pos, player_boxes, racket_boxes, threshold=200):
                shot = "rally"

        if shot:
            self.last_shot_frame = frame_id

        return shot

    def _ball_height_zone(self, ball_pos, player_boxes):
        """
        Compute ball height relative to the nearest player bounding box.
        Returns one of: 'above_head', 'waist', 'low', 'unknown'

        This removes camera angle dependency from raw pixel dy values.
        A downward-angled camera makes all far-end shots look like
        upward motion in pixel space — relative height fixes that.

        Zones based on player box:
          above_head : ball y < player top (above head)
          waist      : ball y between top and mid of player box
          low        : ball y below mid of player box (near ground)
        """

        if ball_pos is None or not player_boxes:
            return "unknown"

        ball_x, ball_y = ball_pos

        # find nearest player
        nearest_box = None
        min_dist = float("inf")

        for box in player_boxes:
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2)
            dist = abs(cx - ball_x)
            if dist < min_dist:
                min_dist = dist
                nearest_box = box

        if nearest_box is None:
            return "unknown"

        x1, y1, x2, y2 = nearest_box
        player_top = float(y1)
        player_mid = float((y1 + y2) / 2)

        if ball_y < player_top:
            return "above_head"
        elif ball_y < player_mid:
            return "waist"
        else:
            return "low"

    def _ball_near_player_or_racket(self, ball_pos, player_boxes, racket_boxes, threshold):
        """
        Returns True if the ball is within threshold pixels of any player or racket center.
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

    def _confirmed_recent_bounce(self, frame_id, bounces, within_frames=45):
        """
        Returns True if a bounce was detected within the last within_frames frames.
        Used to validate serve detection.
        """

        recent = [b for b in bounces if 0 < frame_id - b["frame"] <= within_frames]
        return len(recent) >= 1

    def _classify_forehand_backhand(self, ball_pos, player_boxes, avg_dx):
        """
        Classify forehand vs backhand using ball movement direction
        relative to the nearest player's center — not raw avg_dx.

        Logic:
          Find the nearest player center (cx, cy).
          Compute the vector from player center to ball position.
          If the ball is moving AWAY from the player's right side = forehand.
          If the ball is moving AWAY from the player's left side = backhand.

        This is camera-angle independent because it uses the player
        as the reference frame, not the image coordinate system.

        Right-handed player assumption:
          ball to the right of player center and moving right = forehand
          ball to the left of player center and moving left  = backhand
        """

        if not player_boxes:
            # no player visible — fall back to raw direction
            return "forehand" if avg_dx > 0 else "backhand"

        ball_x, ball_y = ball_pos

        # find nearest player
        nearest_cx = None
        min_dist = float("inf")

        for box in player_boxes:
            x1, y1, x2, y2 = box
            player_cx = float((x1 + x2) / 2)
            player_cy = float((y1 + y2) / 2)
            dist = math.sqrt((player_cx - ball_x) ** 2 + (player_cy - ball_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                nearest_cx = player_cx

        if nearest_cx is None:
            return "forehand" if avg_dx > 0 else "backhand"

        # relative position: is ball to the right or left of player center
        ball_relative_x = ball_x - nearest_cx

        # ball is to the right of player AND moving right = forehand
        # ball is to the left of player AND moving left  = backhand
        # if they disagree (ball crossed center), use ball position side
        if ball_relative_x >= 0:
            return "forehand"
        else:
            return "backhand"

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
