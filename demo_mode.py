from src.pipeline import Pipeline


def run_demo():
    print("🎮 Running Demo Mode...")

    pipeline = Pipeline(
        video_path="data/input_sample_video.mp4",
        model_path="models/yolov8n.pt",
        output_dir="results"
    )

    pipeline.run()


if __name__ == "__main__":
    run_demo()