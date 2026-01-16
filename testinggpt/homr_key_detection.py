#!/usr/bin/env python3
"""
Use HOMR to detect staffs and measures, extract first measures, and detect key signatures with GPT
"""
import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Add homr_repo to path
sys.path.insert(0, str(Path(__file__).parent.parent / "homr_repo"))

from homr.main import detect_staffs_in_image, ProcessingConfig, load_and_preprocess_predictions, predict_symbols
from homr.bar_line_detection import detect_bar_lines
from homr.note_detection import combine_noteheads_with_stems
from homr.debug import Debug

load_dotenv(Path(__file__).parent.parent / "homr_repo" / ".env")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def detect_key_signature_gpt(image_path, model="gpt-4o", verbose=True):
    """
    Detect key signature from a single staff first measure image using GPT

    Args:
        image_path: Path to first measure image
        model: GPT model to use (default: gpt-4o)
        verbose: Print progress messages

    Returns:
        str: Key signature (e.g., "G", "C", "F") or None on error
    """
    if verbose:
        print(f"  Analyzing with GPT: {Path(image_path).name}")

    # Encode
    base64_image = encode_image(image_path)

    prompt = """Look at this single staff from sheet music.

CRITICAL: Look at the area RIGHT AFTER the clef symbol on the LEFT side.

INSTRUCTIONS:
1. Identify the clef symbol (treble or bass)
2. Count ONLY the sharps (#) or flats (♭) immediately after the clef (this is the key signature)
3. IGNORE any accidentals in the middle of the measure
4. Map to key name using the chart below

KEY SIGNATURE CHART:
- 0 sharps/flats = C
- 1 sharp = G
- 2 sharps = D
- 3 sharps = A
- 4 sharps = E
- 5 sharps = B
- 6 sharps = F#
- 1 flat = F
- 2 flats = Bb
- 3 flats = Eb
- 4 flats = Ab
- 5 flats = Db
- 6 flats = Gb

RESPONSE: Return ONLY the key signature letter(s) (e.g., "G" or "C" or "Bb"), nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=800
        )

        # Parse response
        raw_response = response.choices[0].message.content.strip()

        if verbose:
            print(f"    GPT Response: {raw_response}")

        # Extract key signature
        key_sig = raw_response.strip('"\'` \n')

        if key_sig:
            if verbose:
                print(f"    ✓ Detected key: {key_sig}")
            return key_sig
        else:
            if verbose:
                print("    ✗ Error: Empty response")
            return None

    except Exception as e:
        if verbose:
            print(f"    ✗ Error: {e}")
        return None


def process_image_with_homr(image_path, output_folder="testinggpt/results"):
    """
    Process image with HOMR to detect staffs and measures, extract first measures, and detect key signatures

    Args:
        image_path: Path to sheet music image
        output_folder: Where to save results
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("HOMR + GPT KEY SIGNATURE DETECTION")
    print(f"{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}\n")

    # Step 1: Run HOMR detection
    print("Step 1: Running HOMR detection (this may take a minute)...")

    try:
        # Load and preprocess
        predictions, debug = load_and_preprocess_predictions(
            str(image_path),
            enable_debug=False,
            enable_cache=False,
            use_gpu_inference=False
        )

        # Extract symbols
        symbols = predict_symbols(debug, predictions)

        # Get bar lines
        noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
        all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
        all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
        bar_lines_or_rests = [
            line
            for line in symbols.bar_lines
            if not line.is_overlapping_with_any(all_noteheads)
            and not line.is_overlapping_with_any(all_stems)
        ]
        average_note_head_height = float(
            np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
        ) if noteheads_with_stems else 10.0
        bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)

        # Run full detection to get staffs
        config = ProcessingConfig(
            enable_debug=False,
            enable_cache=False,
            write_staff_positions=False,
            read_staff_positions=False,
            selected_staff=-1,
            use_gpu_inference=False,
            visualize=False
        )
        multi_staffs, preprocessed, debug, title_future, notes, staffs, original = detect_staffs_in_image(
            str(image_path),
            config
        )

    except Exception as e:
        print(f"✗ Error running HOMR: {e}")
        import traceback
        traceback.print_exc()
        return

    if not staffs:
        print("✗ Error: No staffs detected!")
        return

    print(f"  ✓ Found {len(staffs)} staff(s)")
    print(f"  ✓ Found {len(bar_line_boxes)} bar line(s)")

    # Step 2: Extract first measures from each staff
    print("\nStep 2: Extracting first measures...")
    results = []

    for staff_idx, staff in enumerate(staffs):
        print(f"\n  Staff {staff_idx + 1}:")

        # Get staff boundaries
        staff_top = int(staff.min_y - staff.average_unit_size * 2)
        staff_bottom = int(staff.max_y + staff.average_unit_size * 2)
        staff_left = int(staff.min_x)
        staff_right = int(staff.max_x)

        # Get bar lines for this staff (within staff y bounds)
        staff_bar_lines = [bl for bl in bar_line_boxes if staff_top <= bl.center[1] <= staff_bottom]

        # Sort bar lines by x position
        staff_bar_lines.sort(key=lambda bl: bl.center[0])

        print(f"    Staff bounds: y={staff_top}-{staff_bottom}, x={staff_left}-{staff_right}")
        print(f"    Found {len(staff_bar_lines)} bar line(s) for this staff")

        # Determine first measure boundaries
        left_x = staff_left
        if staff_bar_lines:
            # First measure is from staff start to first bar line
            right_x = int(staff_bar_lines[0].center[0])
            print(f"    First measure: x={left_x} to x={right_x}")
        else:
            # No bar lines, use whole staff width
            right_x = staff_right
            print(f"    No bar lines found, using full staff width")

        # Make sure we have valid bounds
        left_x = max(0, left_x)
        right_x = min(original.shape[1], right_x)
        staff_top = max(0, staff_top)
        staff_bottom = min(original.shape[0], staff_bottom)

        # Extract first measure image from original
        first_measure_img = original[staff_top:staff_bottom, left_x:right_x].copy()

        # Find clefs within this first measure
        measure_clefs = []
        for clef_box in symbols.clefs_keys:
            clef_center_x, clef_center_y = clef_box.center
            # Check if clef is within this first measure bounds
            if (left_x <= clef_center_x <= right_x and
                staff_top <= clef_center_y <= staff_bottom):
                measure_clefs.append(clef_box)

        print(f"    Found {len(measure_clefs)} clef(s) in first measure")

        # Find the biggest clef (by area)
        if measure_clefs:
            biggest_clef = max(measure_clefs, key=lambda c: c.size[0] * c.size[1])
            print(f"    Biggest clef size: {biggest_clef.size[0]:.1f}x{biggest_clef.size[1]:.1f}")
            measure_clefs = [biggest_clef]  # Only keep the biggest one

        # Draw the biggest clef on the image
        for clef in measure_clefs:
            # Adjust coordinates relative to the cropped image
            clef_x = int(clef.center[0] - left_x)
            clef_y = int(clef.center[1] - staff_top)

            # Draw bounding box around clef
            box_points = cv2.boxPoints(clef.box).astype(np.int32)
            adjusted_points = []
            for pt in box_points:
                adjusted_pt = (int(pt[0] - left_x), int(pt[1] - staff_top))
                adjusted_points.append(adjusted_pt)

            # Draw the rotated rectangle
            adjusted_points_np = np.array(adjusted_points, dtype=np.int32)
            cv2.polylines(first_measure_img, [adjusted_points_np], True, (0, 255, 0), 3)

            # Draw center point
            cv2.circle(first_measure_img, (clef_x, clef_y), 5, (255, 0, 0), -1)

            # Add label
            cv2.putText(first_measure_img, "CLEF", (clef_x - 30, clef_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save first measure image with clef annotations
        measure_path = output_path / f"staff_{staff_idx + 1}_first_measure.png"
        cv2.imwrite(str(measure_path), first_measure_img)
        print(f"    ✓ Saved: {measure_path}")

        # Step 3: Detect key signature with GPT
        print(f"    Detecting key signature with GPT...")
        key_sig = detect_key_signature_gpt(str(measure_path), verbose=True)

        results.append({
            'staff_number': staff_idx + 1,
            'filename': measure_path.name,
            'success': key_sig is not None,
            'key_signature': key_sig,
            'staff_bounds': {'top': staff_top, 'bottom': staff_bottom, 'left': staff_left, 'right': staff_right},
            'measure_bounds': {'left': left_x, 'right': right_x},
            'timestamp': datetime.now().isoformat()
        })

    # Save results
    results_file = output_path / f"key_signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"✓ Results saved to: {results_file}\n")

    # Summary
    print("SUMMARY:")
    for result in results:
        if result['success']:
            print(f"  ✓ Staff {result['staff_number']}: {result['key_signature']} major")
        else:
            print(f"  ✗ Staff {result['staff_number']}: FAILED")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect key signatures using HOMR + GPT-4o'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input image file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='testinggpt/results',
        help='Output folder for results'
    )

    args = parser.parse_args()

    process_image_with_homr(args.input, args.output)
