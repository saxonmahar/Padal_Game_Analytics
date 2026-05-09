import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter


class Analytics:
    """Collects per-frame stats and saves all result files at the end."""
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

                if cls == 0:    # person
                    person_count += 1
                elif cls == 38: # tennis racket
                    racket_count += 1

        self.total_players_detected += person_count

        frame_info = {
            "frame_id": frame_id,
            "players_detected": person_count,
            "rackets_detected": racket_count
        }

        self.frame_data.append(frame_info)

        return frame_info

    def save_results(self, output_dir, shots=None, bounces=None):
        """
        Save all results:
        - shots_detected.json  : full shot list with frame, timestamp, type, player
        - shots.csv            : same data in CSV format
        - frame_data.json      : per-frame player/racket counts
        - summary.json         : totals and shot type breakdown
        - dashboard.png        : full multi-chart visual dashboard
        """

        os.makedirs(output_dir, exist_ok=True)

        shots = shots or []
        bounces = bounces or []

        shots_path = os.path.join(output_dir, "shots_detected.json")
        with open(shots_path, "w") as f:
            json.dump(shots, f, indent=4)

        csv_path = os.path.join(output_dir, "shots.csv")
        if shots:
            df_shots = pd.DataFrame(shots)
        else:
            df_shots = pd.DataFrame(columns=["frame_id", "timestamp_seconds", "shot_type", "player_id"])
        df_shots.to_csv(csv_path, index=False)

        frame_json_path = os.path.join(output_dir, "frame_data.json")
        with open(frame_json_path, "w") as f:
            json.dump(self.frame_data, f, indent=4)

        shot_counts = dict(Counter(s["shot_type"] for s in shots))
        summary = {
            "total_frames": self.total_frames,
            "total_player_detections": self.total_players_detected,
            "avg_players_per_frame": round(
                self.total_players_detected / self.total_frames, 2
            ) if self.total_frames > 0 else 0,
            "total_shots": len(shots),
            "total_bounces": len(bounces),
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
        self._generate_dashboard(output_dir, df_frames, shots, bounces)

    def _generate_dashboard(self, output_dir, df, shots, bounces):
        """
        Generate a recruiter-friendly multi-chart dashboard:
        - Shot distribution bar chart
        - Shot type pie chart
        - Shot timeline (scatter)
        - Player detection histogram
        - Bounce count summary
        """

        shot_types = [s["shot_type"] for s in shots]
        shot_counts = Counter(shot_types)
        labels = list(shot_counts.keys())
        values = list(shot_counts.values())
        total_shots = len(shots)
        total_bounces = len(bounces)

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"Padel Analytics Dashboard   |   Total Shots: {total_shots}   |   Total Bounces: {total_bounces}",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # --------------------------------------------------
        # Chart 1: Shot Distribution Bar Chart
        # --------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])

        if labels:
            bars = ax1.bar(labels, values, color=colors[:len(labels)], edgecolor="white", linewidth=0.8)
            ax1.bar_label(bars, fmt="%d", padding=3, fontsize=10)
        else:
            ax1.text(0.5, 0.5, "No shots detected", ha="center", va="center", transform=ax1.transAxes)

        ax1.set_title("Shot Count by Type", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Shot Type")
        ax1.set_ylabel("Count")
        ax1.set_ylim(0, max(values) * 1.2 if values else 1)

        # --------------------------------------------------
        # Chart 2: Shot Type Pie Chart
        # --------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])

        if labels:
            wedges, texts, autotexts = ax2.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors[:len(labels)],
                startangle=140,
                pctdistance=0.82
            )
            for at in autotexts:
                at.set_fontsize(9)
        else:
            ax2.text(0.5, 0.5, "No shots detected", ha="center", va="center", transform=ax2.transAxes)

        ax2.set_title("Shot Type Distribution (%)", fontsize=12, fontweight="bold")

        # --------------------------------------------------
        # Chart 3: Shot Timeline (when each shot happened)
        # --------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])

        if shots:
            shot_type_list = [s["shot_type"] for s in shots]
            timestamps = [s["timestamp_seconds"] for s in shots]
            unique_types = list(set(shot_type_list))
            type_color_map = {t: colors[i % len(colors)] for i, t in enumerate(unique_types)}

            for shot in shots:
                ax3.scatter(
                    shot["timestamp_seconds"],
                    shot["shot_type"],
                    color=type_color_map[shot["shot_type"]],
                    s=60,
                    zorder=3
                )

            ax3.set_title("Shot Timeline", fontsize=12, fontweight="bold")
            ax3.set_xlabel("Time (seconds)")
            ax3.set_ylabel("Shot Type")
            ax3.grid(axis="x", linestyle="--", alpha=0.5)
        else:
            ax3.text(0.5, 0.5, "No shots detected", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Shot Timeline", fontsize=12, fontweight="bold")

        # --------------------------------------------------
        # Chart 4: Players Detected per Frame (line)
        # --------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 0:2])

        ax4.plot(df["frame_id"], df["players_detected"], color="#4C72B0", label="Players", linewidth=1.2)
        ax4.plot(df["frame_id"], df["rackets_detected"], color="#DD8452", label="Rackets", linewidth=1.2, linestyle="--")

        # mark bounce frames on the timeline
        if bounces:
            bounce_frames = [b["frame"] for b in bounces]
            ax4.vlines(bounce_frames, ymin=0, ymax=df["players_detected"].max(),
                       colors="red", linewidth=0.8, alpha=0.5, label="Bounce")

        ax4.set_title("Detections per Frame (Players, Rackets, Bounces)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Frame")
        ax4.set_ylabel("Count")
        ax4.legend(loc="upper right")
        ax4.grid(axis="y", linestyle="--", alpha=0.4)

        # --------------------------------------------------
        # Chart 5: Summary Stats (text panel)
        # --------------------------------------------------
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")

        avg_players = round(
            df["players_detected"].mean(), 2
        ) if not df.empty else 0

        summary_lines = [
            f"Total Frames:      {len(df)}",
            f"Total Shots:       {total_shots}",
            f"Total Bounces:     {total_bounces}",
            f"Avg Players/Frame: {avg_players}",
            "",
        ]

        for shot_type, count in shot_counts.items():
            pct = round(count / total_shots * 100, 1) if total_shots > 0 else 0
            summary_lines.append(f"  {shot_type:<12} {count:>4}  ({pct}%)")

        summary_text = "\n".join(summary_lines)

        ax5.text(
            0.05, 0.95,
            summary_text,
            transform=ax5.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc")
        )

        ax5.set_title("Summary", fontsize=12, fontweight="bold")

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        dashboard_path = os.path.join(output_dir, "dashboard.png")
        plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  dashboard.png       -> {dashboard_path}")
