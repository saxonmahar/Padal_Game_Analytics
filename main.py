import os
from src.pipeline import Pipeline


def main():
    video_path = "data/input_sample_video.mp4"
    model_path = "models/yolov8n.pt"
    output_dir = "results"

    os.makedirs(output_dir, exist_ok=True)

    print("Starting Padel Analytics Pipeline...")

    pipeline = Pipeline(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir
    )

    pipeline.run()

    print("Processing completed.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
