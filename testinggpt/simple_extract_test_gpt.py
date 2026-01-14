#!/usr/bin/env python3
"""
Simple first measure extraction for GPT testing
Uses basic CV to extract first measures, then tests GPT key signature detection
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from detect_key_signatures import detect_key_signatures


def simple_extract_first_measures(image_path, output_dir="first_measures_test"):
    """
    Simple extraction: crop left portion of each detected staff region
    """
    print(f"Processing: {image_path}")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print("❌ Could not load image")
        return None

    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")

    # For your clean input image, manually define approximate staff regions
    # Based on typical sheet music layout - adjust if needed
    num_staffs = 12  # Your image has 12 staffs
    staff_height_approx = height // num_staffs

    # Extract first ~400 pixels (first measure area) for each staff
    first_measure_width = 400

    first_measures = []

    print("\nExtracting first measures...")
    for i in range(num_staffs):
        # Calculate staff region
        staff_top = i * staff_height_approx
        staff_bottom = (i + 1) * staff_height_approx

        # Crop first measure area
        cropped = img[staff_top:staff_bottom, 0:first_measure_width].copy()

        # Find actual content bounds to add red lines at staff positions
        # Approximate staff line positions (5 evenly spaced lines in middle 60% of region)
        staff_region_height = staff_bottom - staff_top
        staff_zone_start = int(staff_region_height * 0.2)
        staff_zone_end = int(staff_region_height * 0.8)
        staff_zone_height = staff_zone_end - staff_zone_start

        # Top and bottom staff lines
        top_line_y = staff_zone_start
        bottom_line_y = staff_zone_end

        # Draw red lines
        cv2.line(cropped, (0, top_line_y), (cropped.shape[1], top_line_y), (0, 0, 255), 3)
        cv2.line(cropped, (0, bottom_line_y), (cropped.shape[1], bottom_line_y), (0, 0, 255), 3)

        # Save
        output_path = output_dir / f"staff_{i+1}_first_measure.png"
        cv2.imwrite(str(output_path), cropped)

        print(f"Staff {i+1}: Saved to {output_path.name}")

        first_measures.append({
            'staff_idx': i + 1,
            'image_path': str(output_path)
        })

    print(f"\n✓ Extracted {len(first_measures)} first measures")
    return first_measures


def test_gpt_on_measures(first_measures, output_dir="first_measures_test"):
    """
    Test GPT key signature detection on extracted measures
    """
    print("\n" + "="*60)
    print("Testing GPT Key Signature Detection")
    print("="*60)

    results = []

    for measure_info in first_measures:
        staff_idx = measure_info['staff_idx']
        image_path = measure_info['image_path']

        print(f"\nStaff {staff_idx}...")

        # Run GPT detection
        key_sigs = detect_key_signatures(image_path, verbose=True)

        if key_sigs:
            # Take first result
            detected_key = key_sigs[0] if isinstance(key_sigs, list) and len(key_sigs) > 0 else key_sigs

            results.append({
                'staff': staff_idx,
                'key_signature': detected_key,
                'success': True
            })
        else:
            results.append({
                'staff': staff_idx,
                'key_signature': None,
                'success': False
            })

        print("-" * 60)

    # Save results
    output_dir = Path(output_dir)
    results_file = output_dir / f"gpt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Summary
    print("\nSUMMARY:")
    for result in results:
        if result['success']:
            print(f"  Staff {result['staff']}: {result['key_signature']}")
        else:
            print(f"  Staff {result['staff']}: FAILED")

    successful = sum(1 for r in results if r['success'])
    print(f"\nSuccess rate: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple first measure extraction + GPT testing")
    parser.add_argument("image_path", help="Path to sheet music image")
    parser.add_argument("-o", "--output_dir", default="first_measures_test", help="Output directory")

    args = parser.parse_args()

    # Step 1: Extract first measures
    first_measures = simple_extract_first_measures(args.image_path, args.output_dir)

    if first_measures:
        # Step 2: Test GPT
        results = test_gpt_on_measures(first_measures, args.output_dir)
