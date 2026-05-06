import math


class Tracker:
    def __init__(self):
        print("Tracker initialized (enhanced)")

        self.ball_history = []
        self.last_position = None

        # smoothing factor (0 = no smoothing, closer to 1 = smoother)
        self.alpha = 0.6

        # missing frame handling
        self.max_missing_frames = 8
        self.missing_count = 0

        # racket tracking: track_id -> list of positions
        self.racket_history = {}

    def update(self, frame_id, ball_pos):
        """
        Improved ball tracking with:
        - smoothing (EMA)
        - missing frame handling
        - velocity calculation

        ball_pos: (cx, cy) from Detector.detect_with_ball_fallback(), or None
        """

        current_pos = ball_pos

        # handle missing ball detection
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

        # apply smoothing (EMA)
        if self.last_position:
            prev_x, prev_y = self.last_position
            curr_x, curr_y = current_pos

            smooth_x = self.alpha * prev_x + (1 - self.alpha) * curr_x
            smooth_y = self.alpha * prev_y + (1 - self.alpha) * curr_y

            current_pos = (smooth_x, smooth_y)

        # compute velocity
        speed = 0
        dx, dy = 0, 0

        if self.last_position:
            prev_x, prev_y = self.last_position
            curr_x, curr_y = current_pos

            dx = curr_x - prev_x
            dy = curr_y - prev_y
            speed = math.sqrt(dx**2 + dy**2)

        # store data
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

    def update_rackets(self, frame_id, rackets):
        """
        Track racket positions per track_id across frames.
        rackets: list of dicts from Detector.detect_rackets()
        """

        for racket in rackets:
            track_id = racket["track_id"]

            # use cx/cy as key if no track_id assigned
            key = track_id if track_id is not None else f"untracked_{frame_id}"

            if key not in self.racket_history:
                self.racket_history[key] = []

            self.racket_history[key].append({
                "frame": frame_id,
                "cx": racket["cx"],
                "cy": racket["cy"],
                "conf": racket["conf"]
            })

    def get_rackets_at_frame(self, frame_id):
        """
        Returns list of racket positions seen at a specific frame.
        """

        rackets_at_frame = []

        for track_id, history in self.racket_history.items():
            for entry in history:
                if entry["frame"] == frame_id:
                    rackets_at_frame.append({
                        "track_id": track_id,
                        "cx": entry["cx"],
                        "cy": entry["cy"]
                    })

        return rackets_at_frame

    def get_ball_trajectory(self):
        return self.ball_history

    def get_racket_history(self):
        return self.racket_history

    def get_ball_speed(self):
        if len(self.ball_history) < 1:
            return 0

        return self.ball_history[-1]["velocity"]["speed"]
