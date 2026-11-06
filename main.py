"""
WeatherSense-VAD — Main Entry Point

Weather-Robust, Context-Aware and Continually Adaptive
Video Anomaly Detection System

Usage:
    python main.py                  # Launch dashboard (default)
    python main.py --demo           # Run demo analysis and print results
    python main.py --analyze IMG    # Analyze a specific image
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="WeatherSense-VAD: Video Anomaly Detection System"
    )
    parser.add_argument(
        "--dashboard", action="store_true", default=True,
        help="Launch the web dashboard (default)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Dashboard port (default: 5000)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo analysis on sample data"
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Analyze a specific image file"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.analyze:
        run_analysis(args.analyze)
    else:
        launch_dashboard(args.port)


def launch_dashboard(port=5000):
    """Launch the Flask dashboard."""
    print("=" * 60)
    print("  WeatherSense-VAD Dashboard")
    print("  Weather-Robust Video Anomaly Detection")
    print("=" * 60)

    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from dashboard.app import app
    print(f"\n  Open http://127.0.0.1:{port} in your browser\n")
    app.run(host="127.0.0.1", port=port, debug=False)


def run_demo():
    """Run a quick demo analysis."""
    import numpy as np

    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from dashboard.app import (
        assess_weather, enhance_image, classify_scene,
        get_temporal_context, compute_anomaly_score,
        generate_explanation, get_clip_scores_from_baseline,
    )

    # Load a sample image
    img_dir = project_root / "data" / "shanghaitech" / "part_A" / "test_data" / "images"
    if img_dir.exists():
        from PIL import Image
        jpgs = sorted(img_dir.glob("*.jpg"))
        if jpgs:
            image = Image.open(jpgs[len(jpgs)//2]).convert("RGB")
            image_np = np.asarray(image).astype(np.float32) / 255.0
            print(f"Loaded: {jpgs[len(jpgs)//2].name}")
        else:
            print("No images found, using synthetic data.")
            image_np = np.random.rand(480, 640, 3).astype(np.float32)
    else:
        print("Dataset not found, using synthetic data.")
        image_np = np.random.rand(480, 640, 3).astype(np.float32)

    print("\n--- Weather Assessment ---")
    weather = assess_weather(image_np)
    for k, v in weather.items():
        print(f"  {k}: {v}")

    print("\n--- Enhancement ---")
    enhanced = enhance_image(image_np, weather)
    print(f"  Applied: {weather['applied_enhancement']}")

    print("\n--- Scene Classification ---")
    scene = classify_scene(enhanced)
    print(f"  Top scene: {scene['top_scene']} ({scene['confidence']:.1%})")

    print("\n--- Temporal Context ---")
    temporal = get_temporal_context()
    print(f"  Context: {temporal['time_context']}")
    print(f"  Expected activity: {temporal['expected_activity']:.0%}")

    print("\n--- Anomaly Detection ---")
    anomaly = compute_anomaly_score(enhanced, weather, scene, temporal)
    clips, threshold = get_clip_scores_from_baseline()
    print(f"  Score: {anomaly['score']:.2f}")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Verdict: {'ANOMALY' if anomaly['score'] > threshold else 'NORMAL'}")

    print("\n--- Clip Scores (from baseline) ---")
    for c in clips:
        marker = " *** ANOMALY" if c["is_anomaly"] else ""
        print(f"  Clip {c['clip']:2d}: {c['score']:.4f}{marker}")

    print("\n[OK] Demo complete. Run 'python main.py' for the dashboard.")


def run_analysis(image_path):
    """Analyze a single image."""
    import time
    import numpy as np
    from PIL import Image

    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from dashboard.app import run_full_pipeline

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0

    print(f"Analyzing: {image_path}")
    result = run_full_pipeline(image_np, time.time())

    print(f"\nWeather:  fog={result['weather']['fog_score']:.1%}  "
          f"low_light={result['weather']['low_light_score']:.1%}  "
          f"rain={result['weather']['rain_score']:.1%}")
    print(f"Scene:    {result['scene']['top_scene']} ({result['scene']['confidence']:.0%})")
    print(f"Temporal: {result['temporal']['time_context']}")
    print(f"Score:    {result['anomaly']['score']:.2f} (threshold: {result['anomaly']['threshold']:.2f})")
    print(f"Verdict:  {'ANOMALY' if result['anomaly']['is_anomaly'] else 'NORMAL'}")
    print(f"Time:     {result['system']['processing_time']}")


if __name__ == "__main__":
    main()
