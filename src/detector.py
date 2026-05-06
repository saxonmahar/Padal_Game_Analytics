import cv2
import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)

        # Background subtractor for motion-based ball fallback
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=40,
            detectShadows=False
        )

        # Padel ball size range in pixels (tune per camera distance)
        self.ball_min_radius = 3
        self.ball_max_radius = 25

        # HSV range for yellow-green padel ball
        self.hsv_lower = np.array([20, 80, 80])
        self.hsv_upper = np.array([45, 255, 255])

        # minimum confidence for racket detection (class 38 = tennis racket)
        # set higher than default to reduce false positives on padel rackets
        self.racket_conf_threshold = 0.45

        print("Hybrid detector ready (YOLO + OpenCV fallback)")

    def detect(self, frame):
        """
        Returns YOLO results with ByteTrack tracking for a frame
        """
        return self.model.track(frame, persist=True, verbose=False)

    def detect_with_ball_fallback(self, frame):
        """
        Hybrid ball detection with ByteTrack tracking enabled.
        Using model.track() instead of model() gives persistent track IDs
        for players and rackets across frames.

        1. Try YOLO (class 32 = sports ball)
        2. Fallback to HSV color segmentation
        3. Fallback to background subtraction (motion blob)

        Returns: results, ball_pos (cx, cy) or None, method string or None
        """

        # persist=True keeps ByteTrack state between frames
        # this is what gives consistent player/racket IDs
        results = self.model.track(frame, persist=True, verbose=False)
        result = results[0]

        # -------------------------------------------
        # 1. YOLO ball detection (class 32 = sports ball)
        # -------------------------------------------
        ball_pos = None

        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    ball_pos = (cx, cy)
                    break

        if ball_pos:
            return results, ball_pos, "yolo"

        # -------------------------------------------
        # 2. HSV color fallback
        # -------------------------------------------
        ball_pos = self._detect_ball_hsv(frame)

        if ball_pos:
            return results, ball_pos, "hsv"

        # -------------------------------------------
        # 3. Motion / background subtraction fallback
        # -------------------------------------------
        ball_pos = self._detect_ball_motion(frame)

        if ball_pos:
            return results, ball_pos, "motion"

        return results, None, None

    def detect_rackets(self, result):
        """
        Extract racket detections from YOLO result.
        Uses COCO class 38 (tennis racket) with a higher confidence threshold
        to reduce false positives on padel rackets.

        Returns list of dicts: {cx, cy, x1, y1, x2, y2, conf, track_id}
        """

        rackets = []

        if result.boxes is None:
            return rackets

        boxes = result.boxes

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])

            if cls != 38:
                continue

            conf = float(box.conf[0])

            # skip low confidence detections
            if conf < self.racket_conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

            # tracking ID if available
            track_id = None
            if hasattr(boxes, "id") and boxes.id is not None:
                track_id = int(boxes.id[i])

            rackets.append({
                "cx": cx,
                "cy": cy,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": round(conf, 2),
                "track_id": track_id
            })

        return rackets

    def _detect_ball_hsv(self, frame):
        """
        Detect ball using HSV color mask (yellow-green range)
        """

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._best_ball_contour(contours)

    def _detect_ball_motion(self, frame):
        """
        Detect ball using background subtraction (catches fast-moving blobs)
        """

        fg_mask = self.bg_subtractor.apply(frame)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._best_ball_contour(contours)

    def _best_ball_contour(self, contours):
        """
        Pick the contour that best matches a small circular ball.
        Returns (cx, cy) or None.
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

            # circularity: 1.0 = perfect circle
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            # prefer high circularity + small radius
            score = circularity * (1.0 / (radius + 1))

            if score > best_score:
                best_score = score
                best_pos = (float(cx), float(cy))

        return best_pos
