import cv2


class Visualizer:
    def __init__(self):
        print("🎨 Visualizer initialized")

    def draw(self, frame, result):
        """
        Draw detections on frame
        """

        if result.boxes is None:
            return frame

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            label = f"{cls}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return frame