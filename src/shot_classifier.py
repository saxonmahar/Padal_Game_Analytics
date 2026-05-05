class ShotClassifier:
    def __init__(self):
        print("🎾 Shot Classifier initialized")

        self.ball_history = []
        self.shots = []

    def update(self, frame_id, results):
        """
        Analyze ball/player motion and classify simple shot types (rule-based).
        """

        result = results[0]
        boxes = result.boxes

        ball_detected = False
        players = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                # COCO class mapping (YOLO)
                # 0 = person, 32 = sports ball
                if cls == 0:
                    players += 1

                if cls == 32:
                    ball_detected = True

        # Store frame state
        frame_state = {
            "frame_id": frame_id,
            "ball_detected": ball_detected,
            "players": players
        }

        self.ball_history.append(frame_state)

        # Simple rule-based shot logic (basic version)
        shot_type = self._classify_shot()

        if shot_type:
            self.shots.append({
                "frame_id": frame_id,
                "shot_type": shot_type
            })

        return shot_type

    def _classify_shot(self):
        """
        Very simple heuristic-based classification.
        (Will be improved later using ball speed + trajectory)
        """

        if len(self.ball_history) < 3:
            return None

        last = self.ball_history[-1]
        prev = self.ball_history[-2]

        # Example logic (placeholder)
        if last["ball_detected"] and not prev["ball_detected"]:
            return "serve"

        if last["players"] >= 2 and last["ball_detected"]:
            return "rally"

        return None

    def get_shots(self):
        return self.shots