#!/usr/bin/env python3
"""
Test script for accidental detection with staff positioning.
"""

import sys
import cv2
import os

# Add homr to path
sys.path.insert(0, os.path.dirname(__file__))

from homr.main import load_and_preprocess_predictions, predict_symbols
from homr.staff_detection import detect_staff, break_wide_fragments
from homr.bar_line_detection import detect_bar_lines
from homr.note_detection import combine_noteheads_with_stems, add_notes_to_staffs
from homr.brace_dot_detection import find_braces_brackets_and_grand_staff_lines, prepare_brace_dot_image
from homr.bounding_boxes import create_rotated_bounding_boxes
from homr.debug import Debug
from homr.accidental_detection import detect_and_position_accidentals
import numpy as np


def test_accidentals(image_path: str):
    print(f"\n{'='*60}")
    print(f"Testing accidental detection on: {image_path}")
    print(f"{'='*60}\n")

    # Step 1: Load and preprocess
    print("Step 1: Loading and preprocessing image...")
    predictions, debug = load_and_preprocess_predictions(
        image_path, enable_debug=False, enable_cache=False, use_gpu_inference=False
    )
    print(f"  Image size: {predictions.original.shape}")

    # Step 2: Predict symbols
    print("\nStep 2: Detecting symbols...")
    symbols = predict_symbols(debug, predictions)
    print(f"  Found {len(symbols.noteheads)} noteheads")

    # Step 3: Detect staffs
    print("\nStep 3: Detecting staffs...")
    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)

    if len(noteheads_with_stems) > 0:
        average_note_head_height = float(
            np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
        )
    else:
        average_note_head_height = 16.0

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)

    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    print(f"  Found {len(staffs)} staffs")

    if len(staffs) == 0:
        print("ERROR: No staffs found!")
        return

    # Step 4: Detect and position accidentals
    print("\nStep 4: Detecting accidentals with YOLOv10...")
    model_path = os.path.join(os.path.dirname(__file__), 'homr', 'models', 'accidentals', 'best.pt')
    print(f"  Using model: {model_path}")

    accidentals = detect_and_position_accidentals(
        image=predictions.original,
        staffs=staffs,
        model_path=model_path,
        confidence_threshold=0.3,
    )

    # Step 5: Add notes to staffs for comparison
    print("\nStep 5: Adding notes to staffs...")
    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))
    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )
    print(f"  Found {len(notes)} notes")

    # Step 6: Create visualizations
    print("\nStep 6: Creating visualizations...")

    # Get multi_staffs for visualization
    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)

    # Write notes visualization
    debug_viz = Debug(predictions.original, image_path, True)
    debug_viz.write_notes_visualization(multi_staffs, notes)
    print(f"  Saved: {image_path.replace('.png', '')}_notes.png")

    # Write accidentals visualization
    if len(accidentals) > 0:
        debug_viz.write_accidentals_visualization(multi_staffs, accidentals)
        print(f"  Saved: {image_path.replace('.png', '')}_accidentals.png")

        # Write full visualization (notes + accidentals)
        debug_viz.write_full_visualization(multi_staffs, notes, accidentals)
        print(f"  Saved: {image_path.replace('.png', '')}_full.png")

    # Step 7: Print results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")

    print(f"\nNotes detected: {len(notes)}")
    print(f"Accidentals detected: {len(accidentals)}")

    if len(accidentals) > 0:
        print("\nAccidentals with pitch names:")
        for i, acc in enumerate(sorted(accidentals, key=lambda a: a.center[0])):
            pitch = acc.get_pitch_name()
            print(f"  {i+1}. {acc.accidental_type.value:12} -> {pitch}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to page_1.png
        image_path = os.path.join(os.path.dirname(__file__), '..', 'page_1.png')
    else:
        image_path = sys.argv[1]

    test_accidentals(image_path)
