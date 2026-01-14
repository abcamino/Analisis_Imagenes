"""
Prepare dataset for training.
Organizes images into the required directory structure.

Usage:
    python prepare_dataset.py --source ../data/raw --output ../data/processed
"""

import argparse
import shutil
from pathlib import Path
import json


def prepare_from_raw(source_dir, output_dir, annotations_file=None):
    """
    Prepare dataset from raw images.

    Expected annotations format (JSON):
    {
        "image_name.jpg": {
            "label": "aneurysm" or "normal",
            "boxes": [[x1, y1, x2, y2], ...]  # optional
        }
    }
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Create output directories
    (output_dir / 'normal').mkdir(parents=True, exist_ok=True)
    (output_dir / 'aneurysm').mkdir(parents=True, exist_ok=True)

    # Load annotations if available
    annotations = {}
    if annotations_file and Path(annotations_file).exists():
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

    # Process images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for img_path in source_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Check if we have annotation
        if img_path.name in annotations:
            label = annotations[img_path.name].get('label', 'unknown')
        else:
            # Default: assume all images have aneurysms (based on the report)
            label = 'aneurysm'
            print(f"No annotation for {img_path.name}, assuming 'aneurysm'")

        # Copy to appropriate folder
        if label in ['normal', 'aneurysm']:
            dest = output_dir / label / img_path.name
            shutil.copy2(img_path, dest)
            print(f"Copied {img_path.name} -> {label}/")
        else:
            print(f"Unknown label '{label}' for {img_path.name}, skipping")

    # Summary
    normal_count = len(list((output_dir / 'normal').glob('*')))
    aneurysm_count = len(list((output_dir / 'aneurysm').glob('*')))

    print(f"\nDataset prepared:")
    print(f"  Normal: {normal_count} images")
    print(f"  Aneurysm: {aneurysm_count} images")
    print(f"  Total: {normal_count + aneurysm_count} images")

    if normal_count + aneurysm_count < 100:
        print("\nWARNING: Very few images for training!")
        print("Recommendations:")
        print("  1. Download ADAM Challenge dataset: https://adam.isi.uu.nl/")
        print("  2. Use data augmentation")
        print("  3. Use transfer learning with more epochs")


def create_sample_annotations(source_dir, output_file):
    """Create a sample annotations file for the user to fill in."""
    source_dir = Path(source_dir)

    annotations = {}
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for img_path in source_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            annotations[img_path.name] = {
                "label": "aneurysm",  # or "normal"
                "notes": "Fill in the correct label"
            }

    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Created sample annotations file: {output_file}")
    print("Edit this file to specify the correct labels for each image.")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--source', type=str, default='../data/raw',
                        help='Source directory with raw images')
    parser.add_argument('--output', type=str, default='../data/processed',
                        help='Output directory for processed dataset')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotations JSON file')
    parser.add_argument('--create-annotations', action='store_true',
                        help='Create sample annotations file')
    args = parser.parse_args()

    if args.create_annotations:
        create_sample_annotations(args.source, args.output + '/annotations.json')
    else:
        prepare_from_raw(args.source, args.output, args.annotations)


if __name__ == '__main__':
    main()
