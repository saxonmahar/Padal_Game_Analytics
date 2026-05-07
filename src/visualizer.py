import cv2


class Visualizer:
    """Draws all overlays onto each video frame."""
    def __init__(self):
        print("Visualizer initialized")

        # COCO class names (only ones we need)
        self.class_names = {
            0: "Person",
            32: "Ball"
        }

    def draw(self, frame, result, shot=None, ball_pos=None, ball_method=None, rackets=None, ball_history=None):
        """
        Draw detections + tracking IDs + ball trail + racket boxes + shot label
        """

        if result.boxes is not None:
            boxes = result.boxes

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                # skip ball and racket — drawn separately below
                if cls in (32, 38):
                    continue

                # get class name
                label = self.class_names.get(cls, f"cls_{cls}")

                # tracking ID (if available)
                track_id = None
                if hasattr(boxes, "id") and boxes.id is not None:
                    track_id = int(boxes.id[i])

                if track_id is not None:
                    label = f"{label} ID:{track_id}"

                # color logic
                if cls == 0:  # person
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)

                # draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # draw label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # draw racket boxes (yellow) with confidence score
        if rackets:
            for racket in rackets:
                x1 = int(racket["x1"])
                y1 = int(racket["y1"])
                x2 = int(racket["x2"])
                y2 = int(racket["y2"])
                conf = racket["conf"]
                track_id = racket["track_id"]

                racket_label = f"Racket {conf:.2f}"
                if track_id is not None:
                    racket_label = f"Racket ID:{track_id} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    racket_label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

        # draw ball trail — last 20 positions fading from bright to dim
        if ball_history and len(ball_history) > 1:
            trail = ball_history[-20:]
            for i, entry in enumerate(trail[:-1]):  # skip last — drawn as main circle
                if entry["position"] is None:
                    continue
                tx, ty = int(entry["position"][0]), int(entry["position"][1])
                # opacity increases toward the most recent position
                alpha = (i + 1) / len(trail)
                radius = max(2, int(5 * alpha))
                color_intensity = int(255 * alpha)
                cv2.circle(frame, (tx, ty), radius, (0, color_intensity, color_intensity), -1)

        # draw ball as circle (color shows which method detected it)
        if ball_pos:
            cx, cy = int(ball_pos[0]), int(ball_pos[1])

            method_colors = {
                "yolo":       (0, 255, 255),  # cyan
                "hsv+motion": (0, 255, 0),    # bright green — high confidence
                "hsv":        (0, 165, 255),  # orange
                "motion":     (128, 128, 255), # purple
            }

            ball_color = method_colors.get(ball_method, (255, 255, 255))

            cv2.circle(frame, (cx, cy), 10, ball_color, 2)
            cv2.circle(frame, (cx, cy), 2, ball_color, -1)

            method_label = f"ball[{ball_method}]" if ball_method else "ball"
            cv2.putText(
                frame,
                method_label,
                (cx + 12, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ball_color,
                1
            )

        # draw shot info (top left)
        if shot:
            cv2.putText(
                frame,
                f"SHOT: {shot.upper()}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        return frame
