import math


class ShotClassifier:
    def __init__(self):
        print("Shot Classifier initialized")

        self.shots = []

        # cooldown to avoid duplicate shots on same hit
        self.last_shot_frame = -30
        self.shot_cooldown = 25

        # hit event detection — track speed trend to find the moment
        # of impact (speed spike) rather than classifying every frame
        self.prev_speed = 0
        self.speed_spike_threshold = 8  # minimum speed increase to count as a hit

    def update(self, frame_id, results, ball_history, bounces, fps=30):
        """
        Classifies shots using ball trajectory + player-relative height + proximity.
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

        # update speed trend for next frame
        if ball_history:
            self.prev_speed = ball_history[-1]["velocity"]["speed"]

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
        ball_zone = self._ball_height_zone(ball_pos, player_boxes)

        # find nearest player distance — used across all rules
        nearest_dist = self._nearest_player_distance(ball_pos, player_boxes)

        # hit event detection — only classify at the moment of impact
        # Kalman smooths velocity so spikes are smaller — threshold lowered
        speed_increased = (speed - self.prev_speed) > 3

        # close contact threshold widened — from top-down camera the ball
        # is rarely within 80px of player torso center
        close_contact = nearest_dist is not None and nearest_dist < 200

        if not speed_increased and not close_contact:
            return None

        shot = None
        # Ball above head OR very fast downward motion near player
        # On top-down cameras ball rarely goes above head in pixels
        # so we also allow fast downward + very close to player
        # ---------------------------------------------------
        if speed > 18 and avg_dy > 8:
            if nearest_dist is not None and nearest_dist < 250:
                shot = "smash"

        elif speed > 15 and avg_dy < -8:
            if self._confirmed_recent_bounce(frame_id, bounces, within_frames=60, min_count=2):
                shot = "serve"

        elif speed > 5 and nearest_dist is not None and nearest_dist < 250:
            shot = self._classify_forehand_backhand(ball_pos, player_boxes, avg_dx)

        elif speed > 4 and players >= 2:
            if nearest_dist is not None and nearest_dist < 350:
                shot = "rally"

        if shot:
            self.last_shot_frame = frame_id

        return shot

    def _nearest_player_distance(self, ball_pos, player_boxes):
        """
        Returns the pixel distance from the ball to the nearest player center.
        Returns None if no players detected.
        """

        if ball_pos is None or not player_boxes:
            return None

        bx, by = ball_pos
        min_dist = float("inf")

        for box in player_boxes:
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            dist = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def _ball_height_zone(self, ball_pos, player_boxes):
        """
        Compute ball height relative to the nearest player bounding box.
        Returns: 'above_head', 'waist', 'low', 'unknown'
        """

        if ball_pos is None or not player_boxes:
            return "unknown"

        ball_x, ball_y = ball_pos

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

    def _confirmed_recent_bounce(self, frame_id, bounces, within_frames=60, min_count=2):
        """
        Returns True only if at least min_count bounces were detected
        within the last within_frames frames.
        Requiring 2+ bounces filters out single false positives from
        background subtraction noise.
        """

        recent = [b for b in bounces if 0 < frame_id - b["frame"] <= within_frames]
        return len(recent) >= min_count

    def _classify_forehand_backhand(self, ball_pos, player_boxes, avg_dx):
        """
        Classify forehand vs backhand using ball position relative to
        nearest player center. Camera-angle independent.
        Right-handed player assumption:
          ball right of player center = forehand
          ball left of player center  = backhand
        """

        if not player_boxes:
            return "forehand" if avg_dx > 0 else "backhand"

        ball_x, ball_y = ball_pos
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

        return "forehand" if ball_x >= nearest_cx else "backhand"

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
