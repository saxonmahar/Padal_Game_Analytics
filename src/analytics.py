import json
import os
import pandas as pd


class Analytics:
    def __init__(self):
        print("📊 Analytics module initialized")

        self.frame_data = []
        self.total_frames = 0
        self.total_players_detected = 0

    def process(self, frame_id, results):
        """
        Store analytics per frame
        """

        self.total_frames += 1

        result = results[0]
        boxes = result.boxes

        person_count = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                # COCO class 0 = person
                if cls == 0:
                    person_count += 1

        # accumulate stats
        self.total_players_detected += person_count

        frame_info = {
            "frame_id": frame_id,
            "players_detected": person_count
        }

        self.frame_data.append(frame_info)

        return frame_info

    def save_results(self, output_dir):
        """
        Save analytics to JSON and CSV
        """

        os.makedirs(output_dir, exist_ok=True)

        # JSON output
        json_path = os.path.join(output_dir, "shots.json")
        with open(json_path, "w") as f:
            json.dump(self.frame_data, f, indent=4)

        # CSV output
        csv_path = os.path.join(output_dir, "shots.csv")
        df = pd.DataFrame(self.frame_data)
        df.to_csv(csv_path, index=False)

        # Summary stats (NEW)
        summary = {
            "total_frames": self.total_frames,
            "total_player_detections": self.total_players_detected,
            "avg_players_per_frame": round(
                self.total_players_detected / self.total_frames, 2
            ) if self.total_frames > 0 else 0
        }

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print("📊 Analytics saved:")
        print(f"JSON → {json_path}")
        print(f"CSV  → {csv_path}")
        print(f"Summary → {summary_path}")