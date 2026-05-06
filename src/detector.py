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

        print("Hybrid detector ready (YOLO + OpenCV fallback)")

    def detect(self, frame):
        """
        Returns YOLO results for a frame
        """
        return self.model(frame, verbose=False)

    def detect_with_ball_fallback(self, frame):
        """
        Hybrid ball detection:
        1. Try YOLO (class 32 = sports ball)
        2. Fallback to HSV color segmentation
        3. Fallback to background subtraction (motion blob)

        Returns: results, ball_pos (cx, cy) or None, method string or None
        """

        results = self.model(frame, verbose=False)
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
