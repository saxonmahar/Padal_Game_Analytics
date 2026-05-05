import math


class Tracker:
    def __init__(self):
        print("📦 Tracker initialized")
        self.ball_history = []

    def update(self, frame_id, results):
        """
        Extract ball position and store trajectory
        """

        result = results[0]
        boxes = result.boxes

        ball_pos = None

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                # 32 = sports ball (COCO)
                if cls == 32:
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    ball_pos = (cx, cy)
                    break

        if ball_pos:
            self.ball_history.append({
                "frame": frame_id,
                "position": ball_pos
            })

        return self.ball_history

    def get_ball_trajectory(self):
        return self.ball_history

    def get_ball_speed(self):
        """
        Calculate simple speed between last 2 points
        """

        if len(self.ball_history) < 2:
            return 0

        x1, y1 = self.ball_history[-2]["position"]
        x2, y2 = self.ball_history[-1]["position"]

        dx = x2 - x1
        dy = y2 - y1

        return math.sqrt(dx * dx + dy * dy)