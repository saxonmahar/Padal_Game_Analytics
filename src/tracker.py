import math
import numpy as np


class Tracker:
    """
    Tracks the ball across frames using a 2D Kalman filter.

    State: [x, y, vx, vy] — position and velocity

    Why Kalman instead of simple smoothing:
    - During occlusion it predicts where the ball should be based on velocity,
      rather than just holding the last known position
    - Noisy detections are weighted against the predicted trajectory,
      so false positives have less impact on the track
    """

    def __init__(self):
        print("Tracker initialized (Kalman filter)")

        self.ball_history = []
        self.missing_count = 0
        self.max_missing_frames = 10

        # reject detections that jump more than this many pixels in one frame
        # catches obvious false positives before they reach the Kalman filter
        self.max_jump_distance = 120

        self._init_kalman()

        # racket positions stored per track_id
        self.racket_history = {}

        # bounce detection state
        self.bounces = []
        self.prev_dy = None
        self.bounce_cooldown = 0

    def _init_kalman(self):
        """Set up Kalman filter matrices."""

        # state transition: x += vx, y += vy each frame
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # we only measure position (x, y), not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # process noise — higher allows faster direction changes
        self.Q = np.eye(4, dtype=float) * 4.0

        # measurement noise — lower = trust detections more, higher = smoother
        # 3.0 keeps velocity responsive enough for shot detection
        self.R = np.eye(2, dtype=float) * 3.0

        self.x = None   # state vector [x, y, vx, vy]
        self.P = None   # covariance matrix

    def _kalman_init(self, pos):
        """Start the filter from the first detected position."""
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0

    def _kalman_predict(self):
        """Predict where the ball will be next frame."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.x[0], self.x[1])

    def _kalman_update(self, measurement):
        """Correct the prediction using an actual detection."""
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def update(self, frame_id, ball_pos):
        """
        Update the tracker with a new detection (or None if ball not found).
        Returns the full ball history list.
        """

        # reject detections that jump too far — likely false positives
        if ball_pos is not None and self.x is not None:
            jump = math.sqrt(
                (ball_pos[0] - self.x[0]) ** 2 +
                (ball_pos[1] - self.x[1]) ** 2
            )
            if jump > self.max_jump_distance:
                ball_pos = None

        # initialise filter on first valid detection
        if ball_pos is not None and self.x is None:
            self._kalman_init(ball_pos)

        if ball_pos is None:
            self.missing_count += 1
            if self.x is not None and self.missing_count <= self.max_missing_frames:
                # predict using velocity instead of holding last position
                current_pos = self._kalman_predict()
            else:
                return self.ball_history
        else:
            self.missing_count = 0
            self._kalman_predict()
            self._kalman_update(ball_pos)
            current_pos = (self.x[0], self.x[1])

        dx = float(self.x[2])
        dy = float(self.x[3])
        speed = math.sqrt(dx ** 2 + dy ** 2)

        self.ball_history.append({
            "frame": frame_id,
            "position": current_pos,
            "velocity": {"dx": dx, "dy": dy, "speed": speed}
        })

        self._detect_bounce(frame_id, dy)
        return self.ball_history

    def _detect_bounce(self, frame_id, dy):
        """
        Detect a bounce when vertical velocity flips from downward to upward.
        Kalman-smoothed velocity means smaller thresholds work reliably.
        """

        if self.bounce_cooldown > 0:
            self.bounce_cooldown -= 1
            self.prev_dy = dy
            return

        if self.prev_dy is not None and self.prev_dy > 1.5 and dy < -1.5:
            pos = (float(self.x[0]), float(self.x[1])) if self.x is not None else None
            self.bounces.append({"frame": frame_id, "position": pos})
            self.bounce_cooldown = 10

        self.prev_dy = dy

    def update_rackets(self, frame_id, rackets):
        """Store racket positions per track_id for each frame."""

        for racket in rackets:
            track_id = racket["track_id"]
            key = track_id if track_id is not None else f"untracked_{frame_id}"

            if key not in self.racket_history:
                self.racket_history[key] = []

            self.racket_history[key].append({
                "frame": frame_id,
                "cx": racket["cx"],
                "cy": racket["cy"],
                "conf": racket["conf"]
            })

    def get_ball_trajectory(self):
        return self.ball_history

    def get_racket_history(self):
        return self.racket_history

    def get_bounces(self):
        return self.bounces

    def get_ball_speed(self):
        if not self.ball_history:
            return 0.0
        return self.ball_history[-1]["velocity"]["speed"]
