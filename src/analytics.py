import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class Analytics:
    def __init__(self):
        print("Analytics module initialized")

        self.frame_data = []
        self.total_frames = 0
        self.total_players_detected = 0

    def process(self, frame_id, results):
        """
        Store per-frame analytics (player counts, racket counts)
        """

        self.total_frames += 1

        result = results[0]
        boxes = result.boxes

        person_count = 0
        racket_count = 0

        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])

                if cls == 0:   # person
                    person_count += 1
                elif cls == 38:  # tennis racket
                    racket_count += 1

        self.total_players_detected += person_count

        frame_info = {
            "frame_id": frame_id,
            "players_detected": person_count,
            "rackets_detected": racket_count
        }

        self.frame_data.append(frame_info)

        return frame_info

    def save_results(self, output_dir, shots=None):
        """
        Save all results:
        - shots_detected.json  : full shot list with frame, timestamp, type, player
        - shots.csv            : same data in CSV format
        - frame_data.json      : per-frame player/racket counts
        - summary.json         : totals and shot type breakdown
        - dashboard.png        : matplotlib charts
        """

        os.makedirs(output_dir, exist_ok=True)

        shots = shots or []

        # ---------- SHOTS DETECTED JSON ----------
        shots_path = os.path.join(output_dir, "shots_detected.json")
        with open(shots_path, "w") as f:
            json.dump(shots, f, indent=4)

        # ---------- SHOTS CSV ----------
        csv_path = os.path.join(output_dir, "shots.csv")
        if shots:
            df_shots = pd.DataFrame(shots)
        else:
            df_shots = pd.DataFrame(columns=["frame_id", "timestamp_seconds", "shot_type", "player_id"])
        df_shots.to_csv(csv_path, index=False)

        # ---------- FRAME DATA JSON ----------
        frame_json_path = os.path.join(output_dir, "frame_data.json")
        with open(frame_json_path, "w") as f:
            json.dump(self.frame_data, f, indent=4)

        # ---------- SUMMARY ----------
        shot_counts = dict(Counter(s["shot_type"] for s in shots))

        summary = {
            "total_frames": self.total_frames,
            "total_player_detections": self.total_players_detected,
            "avg_players_per_frame": round(
                self.total_players_detected / self.total_frames, 2
            ) if self.total_frames > 0 else 0,
            "total_shots": len(shots),
            "shot_counts": shot_counts
        }

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print("Analytics saved:")
        print(f"  shots_detected.json -> {shots_path}")
        print(f"  shots.csv           -> {csv_path}")
        print(f"  frame_data.json     -> {frame_json_path}")
        print(f"  summary.json        -> {summary_path}")

        # ---------- DASHBOARD ----------
        df_frames = pd.DataFrame(self.frame_data)
        self._generate_dashboard(output_dir, df_frames, shots)

    def _generate_dashboard(self, output_dir, df, shots):
        """
        Generate matplotlib dashboard
        """

        plt.figure(figsize=(12, 8))

        # plot 1: players per frame
        plt.subplot(2, 1, 1)
        plt.plot(df["frame_id"], df["players_detected"], label="Players")
        plt.plot(df["frame_id"], df["rackets_detected"], label="Rackets", linestyle="--")
        plt.title("Detections per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Count")
        plt.legend()

        # plot 2: shot distribution
        plt.subplot(2, 1, 2)

        if shots:
            shot_types = [s["shot_type"] for s in shots]
            counts = Counter(shot_types)

            labels = list(counts.keys())
            values = list(counts.values())

            plt.bar(labels, values, color="steelblue")
            plt.title("Shot Distribution")
            plt.xlabel("Shot Type")
            plt.ylabel("Count")
        else:
            plt.text(0.5, 0.5, "No shots detected", ha="center")

        total_shots = len(shots)
        plt.suptitle(f"Total Shots: {total_shots}", fontsize=16)

        plt.tight_layout()

        dashboard_path = os.path.join(output_dir, "dashboard.png")
        plt.savefig(dashboard_path)

        print(f"  dashboard.png       -> {dashboard_path}")
