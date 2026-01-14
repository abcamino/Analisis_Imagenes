#!/usr/bin/env python3
"""
Aneurysm Detection System - Main Entry Point

Hybrid Python/C++ implementation for detecting aneurysms in brain CT images.
This is an educational project and should NOT be used for clinical diagnosis.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Aneurysm Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single image
  python main.py --image data/raw/image.jpg

  # Analyze all images in a directory
  python main.py --dir data/raw/

  # Run benchmark
  python main.py --benchmark --image data/raw/image.jpg

  # Export model to ONNX
  python main.py --export-model

NOTE: This is for educational purposes only. NOT for clinical diagnosis.
        """
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory containing images to analyze"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show visualization of results"
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualizations to output directory"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--export-model",
        action="store_true",
        help="Export PyTorch model to ONNX format"
    )
    parser.add_argument(
        "--check-cpp",
        action="store_true",
        help="Check if C++ module is available"
    )

    args = parser.parse_args()

    # Check C++ module
    if args.check_cpp:
        check_cpp_module()
        return

    # Export model
    if args.export_model:
        export_model()
        return

    # Need either --image or --dir
    if not args.image and not args.dir:
        parser.print_help()
        print("\nError: Please provide --image or --dir")
        sys.exit(1)

    # Run detection
    run_detection(args)


def check_cpp_module():
    """Check if C++ module is available."""
    print("Checking C++ module...")
    try:
        import aneurysm_cpp
        print(f"C++ module loaded successfully!")
        print(f"Version: {aneurysm_cpp.__version__}")

        # Test preprocessor
        p = aneurysm_cpp.Preprocessor(224, 224)
        print(f"Preprocessor created: {p.get_target_width()}x{p.get_target_height()}")
        print("C++ module is working correctly!")

    except ImportError as e:
        print(f"C++ module NOT available: {e}")
        print("\nTo build the C++ module:")
        print("  1. Install dependencies: OpenCV, pybind11, CMake")
        print("  2. Run: pip install .")
        print("\nThe system will use Python fallback (slower) until the module is built.")


def export_model():
    """Export model to ONNX format."""
    print("Exporting model to ONNX...")
    try:
        from src.models.export_onnx import export_to_onnx

        output_path = export_to_onnx(
            model_name="mobilenetv3_small_100",
            num_classes=2,
            input_size=(224, 224),
            output_path="models/onnx/mobilenetv3_aneurysm.onnx"
        )
        print(f"Model exported to: {output_path}")

    except ImportError as e:
        print(f"Error: {e}")
        print("Install training dependencies: pip install torch torchvision timm")


def run_detection(args):
    """Run detection pipeline."""
    from src.inference.pipeline import DetectionPipeline
    from src.visualization.overlay import DetectionOverlay

    # Initialize pipeline
    config_path = args.config if Path(args.config).exists() else None
    pipeline = DetectionPipeline(config_path=config_path)

    # Collect images
    if args.image:
        image_paths = [args.image]
    else:
        image_dir = Path(args.dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_paths = [str(p) for p in image_paths]

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    print(f"\nAnalyzing {len(image_paths)} image(s)...")
    print("-" * 50)

    results = []

    # Benchmark mode
    if args.benchmark and len(image_paths) == 1:
        print("Running benchmark (10 iterations)...")
        timings = pipeline.benchmark(image_paths[0], n_runs=10)
        print(f"\nBenchmark results:")
        print(f"  Preprocessing: {timings['preprocess_ms']:.2f}ms")
        print(f"  Inference:     {timings['inference_ms']:.2f}ms")
        print(f"  Postprocess:   {timings['postprocess_ms']:.2f}ms")
        print(f"  Total:         {timings['total_ms']:.2f}ms")
        return

    # Process images
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")

        result = pipeline.run(image_path)
        results.append(result)

        # Print results
        print(f"  Aneurysm detected: {result['has_aneurysm']}")
        print(f"  Confidence: {result['max_confidence']:.1%}")
        print(f"  Time: {result['timings']['total_ms']:.1f}ms")
        print(f"    - Preprocess: {result['timings']['preprocess_ms']:.1f}ms")
        print(f"    - Inference:  {result['timings']['inference_ms']:.1f}ms")

    # Visualization
    if args.visualize or args.save_viz:
        overlay = DetectionOverlay()

        if args.save_viz:
            output_dir = Path(args.output) / "visualizations"
            output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            image_path = result['image_path']

            import cv2
            image = cv2.imread(image_path)

            if args.save_viz:
                save_path = output_dir / f"{Path(image_path).stem}_detection.png"
                overlay.draw_detections_matplotlib(
                    image,
                    result['detections'],
                    save_path=str(save_path)
                )
            elif args.visualize:
                import matplotlib.pyplot as plt
                overlay.draw_detections_matplotlib(image, result['detections'])
                plt.show()

    # Generate report
    if len(results) > 0:
        overlay = DetectionOverlay()
        report_path = overlay.generate_report(results, Path(args.output) / "reports")

    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("NOTE: This is for EDUCATIONAL purposes only.")
    print("DO NOT use for clinical diagnosis.")
    print("=" * 50)


if __name__ == "__main__":
    main()
