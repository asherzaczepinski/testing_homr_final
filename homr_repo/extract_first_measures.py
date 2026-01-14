#!/usr/bin/env python3
"""
Extract first measures from sheet music and highlight staff lines
Finds the rightmost barline that ends the first measure on each line,
crops to that point, and highlights top/bottom staff lines in red
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def detect_staff_lines(image):
    """
    Detect horizontal staff lines in the image
    Returns list of y-coordinates for each staff line
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Use horizontal projection to find staff lines
    horizontal_projection = np.sum(gray < 128, axis=1)

    # Find peaks in the projection (staff lines appear as dark horizontal lines)
    threshold = np.max(horizontal_projection) * 0.3
    staff_line_candidates = []

    for i in range(len(horizontal_projection)):
        if horizontal_projection[i] > threshold:
            staff_line_candidates.append(i)

    # Group nearby lines into staves (5 lines per staff)
    staff_lines = []
    if staff_line_candidates:
        current_group = [staff_line_candidates[0]]

        for i in range(1, len(staff_line_candidates)):
            if staff_line_candidates[i] - staff_line_candidates[i-1] < 5:
                current_group.append(staff_line_candidates[i])
            else:
                if current_group:
                    staff_lines.append(int(np.mean(current_group)))
                current_group = [staff_line_candidates[i]]

        if current_group:
            staff_lines.append(int(np.mean(current_group)))

    return staff_lines


def detect_vertical_lines(image):
    """
    Detect vertical lines (barlines) in the image
    Returns list of x-coordinates
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Use vertical projection to find vertical lines
    vertical_projection = np.sum(gray < 128, axis=0)

    # Find peaks in vertical projection (barlines)
    threshold = np.max(vertical_projection) * 0.4
    barline_candidates = []

    for i in range(len(vertical_projection)):
        if vertical_projection[i] > threshold:
            barline_candidates.append(i)

    # Group nearby pixels into single barlines
    barlines = []
    if barline_candidates:
        current_group = [barline_candidates[0]]

        for i in range(1, len(barline_candidates)):
            if barline_candidates[i] - barline_candidates[i-1] < 10:
                current_group.append(barline_candidates[i])
            else:
                if current_group:
                    barlines.append(int(np.mean(current_group)))
                current_group = [barline_candidates[i]]

        if current_group:
            barlines.append(int(np.mean(current_group)))

    return barlines


def group_staff_lines_into_staves(staff_lines):
    """
    Group individual staff lines into staves (groups of 5)
    Returns list of tuples (top_line, bottom_line) for each staff
    """
    staves = []

    if len(staff_lines) < 5:
        return staves

    i = 0
    while i < len(staff_lines) - 4:
        # Check if next 5 lines form a staff (relatively evenly spaced)
        group = staff_lines[i:i+5]
        spacings = [group[j+1] - group[j] for j in range(4)]
        avg_spacing = np.mean(spacings)

        # If spacings are relatively consistent, it's a staff
        if all(abs(s - avg_spacing) < avg_spacing * 0.5 for s in spacings):
            staves.append((group[0], group[4]))  # top and bottom line
            i += 5
        else:
            i += 1

    return staves


def find_first_measure_end(barlines, image_width):
    """
    Find the rightmost barline that ends the first measure
    Looks for the first significant barline after the clef area
    """
    if not barlines:
        return int(image_width * 0.3)  # Default: 30% of image width

    # Skip barlines in the first 5% (clef area)
    min_x = int(image_width * 0.05)
    valid_barlines = [x for x in barlines if x > min_x]

    if not valid_barlines:
        return int(image_width * 0.3)

    # Return the first barline (which should be at the end of the first measure)
    # Or if we want to be safer, the second barline
    if len(valid_barlines) >= 2:
        return valid_barlines[1]  # Second barline = end of first measure
    else:
        return valid_barlines[0]


def extract_and_highlight(image_path, output_path=None):
    """
    Main function to extract first measures and highlight staff lines
    """
    print(f"Processing: {image_path}")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")

    # Detect staff lines
    print("Detecting staff lines...")
    staff_line_ys = detect_staff_lines(image)
    print(f"Found {len(staff_line_ys)} staff line candidates")

    # Group into staves
    staves = group_staff_lines_into_staves(staff_line_ys)
    print(f"Grouped into {len(staves)} staves")
    for i, (top, bottom) in enumerate(staves):
        print(f"  Staff {i+1}: top={top}, bottom={bottom}")

    # Detect barlines
    print("Detecting barlines...")
    barlines = detect_vertical_lines(image)
    print(f"Found {len(barlines)} barlines at x-positions: {barlines[:10]}...")

    # Find the rightmost barline for first measure
    crop_x = find_first_measure_end(barlines, width)
    print(f"Cropping at x={crop_x} (end of first measure)")

    # Crop image to first measure
    cropped = image[:, :crop_x].copy()
    print(f"Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")

    # Highlight top and bottom staff lines in red for each staff
    print("Highlighting staff lines in red...")
    for i, (top_y, bottom_y) in enumerate(staves):
        # Draw red line on top staff line
        cv2.line(cropped, (0, top_y), (crop_x, top_y), (0, 0, 255), 2)

        # Draw red line on bottom staff line
        cv2.line(cropped, (0, bottom_y), (crop_x, bottom_y), (0, 0, 255), 2)

        print(f"  Highlighted staff {i+1}: top line at y={top_y}, bottom line at y={bottom_y}")

    # Save output
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_first_measures_highlighted.png"

    cv2.imwrite(str(output_path), cropped)
    print(f"✓ Saved to: {output_path}")

    return cropped, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first measures and highlight staff lines")
    parser.add_argument("image_path", help="Path to input sheet music image")
    parser.add_argument("-o", "--output", help="Output path (optional)")

    args = parser.parse_args()

    extract_and_highlight(args.image_path, args.output)
