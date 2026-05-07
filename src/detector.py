import cv2
import math
import numpy as np
from ultralytics import YOLO


class Detector:
    """
    Handles all detection: players, ball, and rackets.

    Ball detection uses three methods in priority order:
      1. YOLO  — most reliable when it works
      2. HSV + motion fusion — requires both color AND movement to agree
      3. HSV or motion alone — last resort fallbacks
    """

    def __init__(self, model_path):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)

        # MOG2 background subtractor — learns what the static court looks like
        # and highlights anything that moves
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=40,
            detectShadows=False
        )

        # expected padel ball size in pixels (depends on camera distance)
        self.ball_min_radius = 3
        self.ball_max_radius = 25

        # yellow-green HSV range for padel ball
        self.hsv_lower = np.array([20, 80, 80])
        self.hsv_upper = np.array([45, 255, 255])

        # COCO class 38 = tennis racket, close enough for padel
        # raised threshold to cut down false positives
        self.racket_conf_threshold = 0.45

        print("Hybrid detector ready (YOLO + OpenCV fallback)")

    def detect(self, frame):
        """Run YOLO with ByteTrack tracking. Returns results list."""
        return self.model.track(frame, persist=True, verbose=False)

    def detect_with_ball_fallback(self, frame):
        """
        Try to find the ball using three methods in order.
        Returns: (results, ball_pos, method_name)
          ball_pos is (cx, cy) or None
          method_name is 'yolo', 'hsv+motion', 'hsv', 'motion', or None
        """

        results = self.model.track(frame, persist=True, verbose=False)
        result = results[0]

        # 1. YOLO — class 32 is sports ball in COCO
        ball_pos = None
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    ball_pos = (float((x1 + x2) / 2), float((y1 + y2) / 2))
                    break

        if ball_pos:
            return results, ball_pos, "yolo"

        # 2. HSV + motion fusion
        # A real ball is yellow-green AND moving.
        # Court markings are yellow-green but static.
        # Player arms move but aren't yellow-green.
        # Both agreeing within 40px = high confidence detection.
        hsv_pos = self._detect_ball_hsv(frame)
        motion_pos = self._detect_ball_motion(frame)

        if hsv_pos and motion_pos:
            dist = math.sqrt(
                (hsv_pos[0] - motion_pos[0]) ** 2 +
                (hsv_pos[1] - motion_pos[1]) ** 2
            )
            if dist < 40:
                return results, hsv_pos, "hsv+motion"

        # 3. Single method fallbacks
        if hsv_pos:
            return results, hsv_pos, "hsv"
        if motion_pos:
            return results, motion_pos, "motion"

        return results, None, None

    def detect_rackets(self, result):
        """
        Find rackets in a YOLO result (class 38 = tennis racket).
        Returns list of dicts with position, confidence, and track ID.
        """

        rackets = []
        if result.boxes is None:
            return rackets

        boxes = result.boxes
        for i, box in enumerate(boxes):
            if int(box.cls[0]) != 38:
                continue

            conf = float(box.conf[0])
            if conf < self.racket_conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            track_id = None
            if hasattr(boxes, "id") and boxes.id is not None:
                track_id = int(boxes.id[i])

            rackets.append({
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "conf": round(conf, 2),
                "track_id": track_id
            })

        return rackets

    def _detect_ball_hsv(self, frame):
        """Find ball-sized circular yellow-green blobs using color thresholding."""

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._best_ball_contour(contours, min_circularity=0.6)

    def _detect_ball_motion(self, frame):
        """Find ball-sized circular moving blobs using background subtraction."""

        fg_mask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # stricter circularity here — player limbs move but aren't circular
        return self._best_ball_contour(contours, min_circularity=0.7)

    def _best_ball_contour(self, contours, min_circularity=0.5):
        """
        From a list of contours, return the center of the one that looks
        most like a small circular ball. Returns (cx, cy) or None.
        """

        best_pos = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (self.ball_min_radius <= radius <= self.ball_max_radius):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # circularity = 1.0 for a perfect circle, lower for irregular shapes
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < min_circularity:
                continue

            # score favors high circularity and small size
            score = circularity * (1.0 / (radius + 1))
            if score > best_score:
                best_score = score
                best_pos = (float(cx), float(cy))

        return best_pos
