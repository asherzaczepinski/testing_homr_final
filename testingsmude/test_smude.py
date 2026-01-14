#!/usr/bin/env python3
"""
Simple test script for smude - Sheet Music Dewarping
Processes all images in test_images/ folder and saves results to output_images/
"""

import os
from pathlib import Path
from skimage.io import imread, imsave
from smude import Smude


def process_images():
    """Process all images in test_images folder"""

    # Setup paths
    test_images_dir = Path("test_images")
    output_dir = Path("output_images")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Check if test_images directory exists and has images
    if not test_images_dir.exists():
        print(f"Error: {test_images_dir} directory not found!")
        return

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in test_images_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {test_images_dir}/")
        print(f"Please add some sheet music images to the test_images/ folder")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print("-" * 50)

    # Initialize Smude
    print("Initializing Smude (this may take a moment on first run)...")
    smude = Smude(use_gpu=False, binarize_output=True)
    print("Smude initialized successfully!")
    print("-" * 50)

    # Process each image
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")

        try:
            # Read image
            image = imread(str(img_file))

            # Process with Smude
            result = smude.process(image)

            # Save result
            output_file = output_dir / f"dewarped_{img_file.stem}.png"
            imsave(str(output_file), result)

            print(f"  ✓ Saved to: {output_file}")

        except Exception as e:
            print(f"  ✗ Error processing {img_file.name}: {e}")

    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Check the {output_dir}/ folder for results")


if __name__ == "__main__":
    process_images()
