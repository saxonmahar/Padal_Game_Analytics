import math


class Tracker:
    def __init__(self):
        print("📦 Tracker initialized (enhanced)")

        self.ball_history = []
        self.last_position = None

        # smoothing factor (0 = no smoothing, closer to 1 = smoother)
        self.alpha = 0.7  

        # missing frame handling
        self.max_missing_frames = 5
        self.missing_count = 0

    def update(self, frame_id, results):
        """
        Improved ball tracking with:
        - smoothing (EMA)
        - missing frame handling
        - velocity calculation
        """

        result = results[0]
        boxes = result.boxes

        current_pos = None

        # 🔍 Detect ball
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                # COCO: 32 = sports ball
                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    current_pos = (cx, cy)
                    break

        # 🧠 Handle missing ball detection
        if current_pos is None:
            self.missing_count += 1

            if self.last_position and self.missing_count <= self.max_missing_frames:
                # reuse last known position
                current_pos = self.last_position
            else:
                # skip if missing too long
                return self.ball_history
        else:
            self.missing_count = 0

        # 🎯 Apply smoothing (EMA)
        if self.last_position:
            prev_x, prev_y = self.last_position
            curr_x, curr_y = current_pos

            smooth_x = self.alpha * prev_x + (1 - self.alpha) * curr_x
            smooth_y = self.alpha * prev_y + (1 - self.alpha) * curr_y

            current_pos = (smooth_x, smooth_y)

        # ⚡ Compute velocity
        speed = 0
        dx, dy = 0, 0

        if self.last_position:
            prev_x, prev_y = self.last_position
            curr_x, curr_y = current_pos

            dx = curr_x - prev_x
            dy = curr_y - prev_y
            speed = math.sqrt(dx**2 + dy**2)

        # 📝 Store data
        frame_data = {
            "frame": frame_id,
            "position": current_pos,
            "velocity": {
                "dx": dx,
                "dy": dy,
                "speed": speed
            }
        }

        self.ball_history.append(frame_data)

        # update last position
        self.last_position = current_pos

        return self.ball_history

    def get_ball_trajectory(self):
        return self.ball_history

    def get_ball_speed(self):
        if len(self.ball_history) < 1:
            return 0

        return self.ball_history[-1]["velocity"]["speed"]