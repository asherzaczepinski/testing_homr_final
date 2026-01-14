#!/usr/bin/env python3
"""
Extract first measures using Orchestra-AI-2 and detect key signatures with GPT
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime

# Add Orchestra-AI-2 to path
orchestra_path = Path(__file__).parent.parent / "Orchestra-AI-2"
sys.path.insert(0, str(orchestra_path))

# Import Orchestra-AI-2 modules
from preprocessing import *
from staff_removal import get_staff_lines, cut_image_into_buckets, get_ref_lines, detect_measure_lines_bounded

# Import our GPT detector
from detect_key_signatures import detect_key_signatures


def group_staff_lines_into_staves(staff_lines):
    """
    Group individual staff lines into staves (groups of 5 consecutive lines).

    Returns:
        List of tuples: [(top_line_idx, bottom_line_idx), ...]
    """
    if len(staff_lines) < 5:
        return []

    staves = []
    i = 0

    while i <= len(staff_lines) - 5:
        # Take 5 consecutive lines as one staff
        group = staff_lines[i:i+5]

        # Check if they're evenly spaced (within tolerance)
        spacings = [group[j+1] - group[j] for j in range(4)]
        avg_spacing = np.mean(spacings)

        # If spacings are relatively consistent, it's a valid staff
        if all(abs(s - avg_spacing) < avg_spacing * 0.7 for s in spacings):
            staves.append((i, i+4))  # Store indices of first and last line
            i += 5
        else:
            i += 1

    return staves


def extract_first_measures_with_orchestra_ai(image_path, output_dir="first_measures_output"):
    """
    Use Orchestra-AI-2 to extract first measures and highlight staff lines.
    """
    print(f"Processing: {image_path}")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load and preprocess image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Could not load image")
        return None, None

    # Preprocess using Orchestra-AI-2 methods
    print("Preprocessing image...")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = img_binary.shape
    print(f"Image size: {width}x{height}")

    # Detect staff lines using Orchestra-AI-2 method
    print("\nDetecting staff lines...")
    staff_lines_thicknesses, staff_lines = get_staff_lines(width, height, img_binary.copy(), threshold=0.8)
    print(f"✓ Detected {len(staff_lines)} staff lines")

    # Group into staves
    staves = group_staff_lines_into_staves(staff_lines)
    print(f"✓ Grouped into {len(staves)} staves")

    for i, (top_idx, bottom_idx) in enumerate(staves):
        print(f"  Staff {i+1}: lines {top_idx}-{bottom_idx} (y={staff_lines[top_idx]}-{staff_lines[bottom_idx]})")

    # Extract first measure for each staff
    # Simplified: just take first 400 pixels (first measure area)
    print("\n--- Extracting First Measures ---")
    first_measures = []

    first_measure_width = 400  # Approximate first measure width

    for staff_idx, (top_line_idx, bottom_line_idx) in enumerate(staves):
        top_y = staff_lines[top_line_idx]
        bottom_y = staff_lines[bottom_line_idx]

        print(f"\nStaff {staff_idx+1}: Extracting first {first_measure_width}px")

        # Add padding
        padding_y = 30

        crop_top = max(0, top_y - padding_y)
        crop_bottom = min(height, bottom_y + padding_y)
        crop_left = 0
        crop_right = min(width, first_measure_width)

        # Crop the first measure (use original color image)
        cropped = img[crop_top:crop_bottom, crop_left:crop_right].copy()

        # Highlight top and bottom staff lines in RED
        # Calculate relative y positions in cropped image
        top_line_y_cropped = top_y - crop_top
        bottom_line_y_cropped = bottom_y - crop_top

        # Draw red horizontal lines
        cv2.line(cropped, (0, top_line_y_cropped), (cropped.shape[1], top_line_y_cropped), (0, 0, 255), 3)
        cv2.line(cropped, (0, bottom_line_y_cropped), (cropped.shape[1], bottom_line_y_cropped), (0, 0, 255), 3)

        print(f"  ✓ Highlighted top line at y={top_y}, bottom line at y={bottom_y}")

        # Save
        output_path = output_dir / f"staff_{staff_idx+1}_first_measure.png"
        cv2.imwrite(str(output_path), cropped)
        print(f"  ✓ Saved: {output_path}")

        first_measures.append({
            'staff_idx': staff_idx + 1,
            'image_path': str(output_path),
            'bounds': {
                'left': int(crop_left),
                'right': int(crop_right),
                'top': int(crop_top),
                'bottom': int(crop_bottom)
            }
        })

    print(f"\n{'='*60}")
    print(f"Extracted {len(first_measures)} first measures")
    print(f"{'='*60}")

    return first_measures, staves


def detect_keys_for_first_measures(first_measures, output_dir="first_measures_output"):
    """
    Run GPT key signature detection on each first measure.
    """
    print("\n--- Running GPT Key Signature Detection ---")

    results = []

    for measure_info in first_measures:
        staff_idx = measure_info['staff_idx']
        image_path = measure_info['image_path']

        print(f"\nStaff {staff_idx}: Detecting key signature...")

        # Run GPT detection
        key_sigs = detect_key_signatures(image_path, verbose=False)

        if key_sigs and len(key_sigs) > 0:
            # Should return just 1 key since it's a single staff
            detected_key = key_sigs[0] if len(key_sigs) == 1 else key_sigs
            print(f"  ✓ Detected: {detected_key}")

            results.append({
                'staff': staff_idx,
                'key_signature': detected_key,
                'image': image_path
            })
        else:
            print(f"  ✗ Detection failed")
            results.append({
                'staff': staff_idx,
                'key_signature': None,
                'image': image_path
            })

    # Save results
    output_dir = Path(output_dir)
    results_file = output_dir / f"key_signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSUMMARY:")
    for result in results:
        if result['key_signature']:
            print(f"  Staff {result['staff']}: {result['key_signature']}")
        else:
            print(f"  Staff {result['staff']}: FAILED")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract first measures and detect key signatures")
    parser.add_argument("image_path", help="Path to sheet music image")
    parser.add_argument("-o", "--output_dir", default="first_measures_output", help="Output directory")

    args = parser.parse_args()

    # Step 1: Extract first measures
    first_measures, staves = extract_first_measures_with_orchestra_ai(args.image_path, args.output_dir)

    if first_measures:
        # Step 2: Detect key signatures
        results = detect_keys_for_first_measures(first_measures, args.output_dir)
