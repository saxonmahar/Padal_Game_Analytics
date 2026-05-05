import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


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

    def save_results(self, output_dir, shots=None):
        """
        Save analytics to JSON, CSV, summary and dashboard
        """

        os.makedirs(output_dir, exist_ok=True)

        # ---------- JSON ----------
        json_path = os.path.join(output_dir, "shots.json")
        with open(json_path, "w") as f:
            json.dump(self.frame_data, f, indent=4)

        # ---------- CSV ----------
        csv_path = os.path.join(output_dir, "shots.csv")
        df = pd.DataFrame(self.frame_data)
        df.to_csv(csv_path, index=False)

        # ---------- SUMMARY ----------
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

        # ---------- DASHBOARD ----------
        self._generate_dashboard(output_dir, df, shots)

    def _generate_dashboard(self, output_dir, df, shots):
        """
        Generate matplotlib dashboard
        """

        plt.figure(figsize=(12, 8))

        # 📈 Plot 1: Players per frame
        plt.subplot(2, 1, 1)
        plt.plot(df["frame_id"], df["players_detected"])
        plt.title("Players Detected per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Players")

        # 📊 Plot 2: Shot distribution
        plt.subplot(2, 1, 2)

        if shots:
            shot_types = [s["shot_type"] for s in shots]
            counts = Counter(shot_types)

            labels = list(counts.keys())
            values = list(counts.values())

            plt.bar(labels, values)
            plt.title("Shot Distribution")
            plt.xlabel("Shot Type")
            plt.ylabel("Count")
        else:
            plt.text(0.5, 0.5, "No shots detected", ha="center")

        # 🔥 Total shots
        total_shots = len(shots) if shots else 0
        plt.suptitle(f"Total Shots: {total_shots}", fontsize=16)

        plt.tight_layout()

        dashboard_path = os.path.join(output_dir, "dashboard.png")
        plt.savefig(dashboard_path)

        print(f"📊 Dashboard saved → {dashboard_path}")