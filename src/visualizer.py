import cv2


class Visualizer:
    def __init__(self):
        print("🎨 Visualizer initialized")

        # COCO class names (only ones we need)
        self.class_names = {
            0: "Person",
            32: "Ball"
        }

    def draw(self, frame, result, shot=None):
        """
        Draw detections + tracking IDs + shot label
        """

        if result.boxes is None:
            return frame

        boxes = result.boxes

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            # Get class name
            label = self.class_names.get(cls, f"cls_{cls}")

            # Tracking ID (if available)
            track_id = None
            if hasattr(boxes, "id") and boxes.id is not None:
                track_id = int(boxes.id[i])

            if track_id is not None:
                label = f"{label} ID:{track_id}"

            # Color logic
            if cls == 0:  # person
                color = (255, 0, 0)
            elif cls == 32:  # ball
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # 🔥 Draw shot info (TOP LEFT)
        if shot:
            cv2.putText(
                frame,
                f"SHOT: {shot}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        return frame