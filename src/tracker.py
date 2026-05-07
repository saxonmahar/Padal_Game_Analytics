import math
import numpy as np


class Tracker:
    """
    Ball tracker using a 2D Kalman filter.

    State vector: [x, y, vx, vy]
      x, y   — ball position
      vx, vy — ball velocity (pixels per frame)

    Why Kalman over EMA:
      EMA just blends old and new positions — it has no model of motion.
      Kalman models the ball as a physical object moving with velocity.
      During occlusion it predicts where the ball should be based on
      its last known velocity, rather than just holding the last position.
      It also weighs noisy detections against the predicted trajectory,
      so false positives have less impact.
    """

    def __init__(self):
        print("Tracker initialized (Kalman filter)")

        self.ball_history = []

        # max pixels the ball can move between frames
        # rejects obvious false detections before feeding to Kalman
        self.max_jump_distance = 120

        # how many consecutive missed frames before we stop predicting
        self.max_missing_frames = 10
        self.missing_count = 0

        # Kalman filter matrices
        # state: [x, y, vx, vy]
        self._init_kalman()

        # racket tracking: track_id -> list of positions
        self.racket_history = {}

        # bounce detection
        self.bounces = []
        self.prev_dy = None
        self.bounce_cooldown = 0

    def _init_kalman(self):
        """
        Initialise Kalman filter matrices.

        State transition (F): position += velocity each frame
        Measurement (H): we observe position only, not velocity
        Process noise (Q): how much we trust the motion model
        Measurement noise (R): how much we trust the detector
        Covariance (P): uncertainty in state estimate

        Tuning:
          Higher R = trust the model more, smooth but laggy
          Lower R  = trust detections more, responsive but noisy
          We use balanced values so the filter tracks real motion
          without being thrown off by occasional false detections.
        """

        # state transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # measurement matrix (we only observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # process noise — how much the ball can deviate from constant velocity
        # higher = allows faster direction changes (good for padel)
        self.Q = np.eye(4, dtype=float) * 4.0

        # measurement noise — how much we trust the detector
        # lowered from 8.0 to 3.0 so Kalman follows detections more closely
        # this keeps velocity estimates responsive enough for shot detection
        self.R = np.eye(2, dtype=float) * 3.0

        # state estimate and covariance — uninitialised
        self.x = None  # state vector [x, y, vx, vy]
        self.P = None  # covariance matrix

    def _kalman_init(self, pos):
        """Initialise state from first detection."""
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0

    def _kalman_predict(self):
        """Predict next state using motion model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (self.x[0], self.x[1])

    def _kalman_update(self, measurement):
        """Update state with a new measurement (x, y)."""
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x                        # innovation
        S = self.H @ self.P @ self.H.T + self.R        # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def update(self, frame_id, ball_pos):
        """
        Update tracker with a new ball detection (or None if not detected).

        Flow:
          1. Reject detections that jump too far (false positives)
          2. If no valid detection: predict using Kalman motion model
          3. If valid detection: Kalman predict + update (fuses model + measurement)
          4. Store smoothed position and velocity
        """

        # reject obvious false positives before feeding to Kalman
        if ball_pos is not None and self.x is not None:
            jump = math.sqrt(
                (ball_pos[0] - self.x[0]) ** 2 +
                (ball_pos[1] - self.x[1]) ** 2
            )
            if jump > self.max_jump_distance:
                ball_pos = None

        # first detection — initialise filter
        if ball_pos is not None and self.x is None:
            self._kalman_init(ball_pos)

        # no detection
        if ball_pos is None:
            self.missing_count += 1

            if self.x is not None and self.missing_count <= self.max_missing_frames:
                # predict position using velocity — better than holding last position
                predicted = self._kalman_predict()
                current_pos = predicted
            else:
                return self.ball_history
        else:
            self.missing_count = 0

            # predict then update (standard Kalman cycle)
            self._kalman_predict()
            self._kalman_update(ball_pos)
            current_pos = (self.x[0], self.x[1])

        # extract velocity from Kalman state
        dx = float(self.x[2])
        dy = float(self.x[3])
        speed = math.sqrt(dx ** 2 + dy ** 2)

        self.ball_history.append({
            "frame": frame_id,
            "position": current_pos,
            "velocity": {
                "dx": dx,
                "dy": dy,
                "speed": speed
            }
        })

        self._detect_bounce(frame_id, dy)

        return self.ball_history

    def _detect_bounce(self, frame_id, dy):
        """
        Bounce detection using Kalman-smoothed velocity.
        dy from Kalman is smoother than raw pixel differences,
        so bounce detection has fewer false positives.
        """

        if self.bounce_cooldown > 0:
            self.bounce_cooldown -= 1
            self.prev_dy = dy
            return

        if self.prev_dy is not None:
            # Kalman velocity is smoother so use smaller thresholds
            if self.prev_dy > 1.5 and dy < -1.5:
                pos = (float(self.x[0]), float(self.x[1])) if self.x is not None else None
                self.bounces.append({
                    "frame": frame_id,
                    "position": pos
                })
                self.bounce_cooldown = 10

        self.prev_dy = dy

    def update_rackets(self, frame_id, rackets):
        """
        Track racket positions per track_id across frames.
        """

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
