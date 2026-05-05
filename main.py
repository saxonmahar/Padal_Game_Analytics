import os
from src.pipeline import Pipeline

def main():
    # Paths (you can later move these to .env)
    video_path = "data/input_sample_video.mp4"
    model_path = "models/yolov8n.pt"
    output_dir = "results"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 Starting Padel Analytics Pipeline...")

    # Initialize pipeline
    pipeline = Pipeline(
        video_path=video_path,
        model_path=model_path,
        output_dir=output_dir
    )

    # Run full processing
    pipeline.run()

    print("✅ Processing completed!")
    print(f"📁 Results saved in: {output_dir}")

if __name__ == "__main__":
    main()