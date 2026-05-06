import cv2


class Visualizer:
    def __init__(self):
        print("Visualizer initialized")

        # COCO class names (only ones we need)
        self.class_names = {
            0: "Person",
            32: "Ball"
        }

    def draw(self, frame, result, shot=None, ball_pos=None, ball_method=None):
        """
        Draw detections + tracking IDs + ball overlay + shot label
        """

        if result.boxes is not None:
            boxes = result.boxes

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                # skip YOLO ball box — drawn separately as circle below
                if cls == 32:
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

        # draw ball as circle (color shows which method detected it)
        if ball_pos:
            cx, cy = int(ball_pos[0]), int(ball_pos[1])

            method_colors = {
                "yolo":   (0, 255, 255),  # cyan
                "hsv":    (0, 165, 255),  # orange
                "motion": (0, 255, 0),    # green
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
