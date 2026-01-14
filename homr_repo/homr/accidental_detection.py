"""
Accidental detection with staff positioning integration.

This module combines YOLOv10 accidental detection with the staff line
positioning system to determine the pitch of each detected accidental.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from homr.model import (
    Accidental,
    AccidentalType,
    Staff,
    note_names,
)
from homr.bounding_boxes import RotatedBoundingBox, BoundingBox
from homr.type_definitions import NDArray

# Try to import the YOLOv10 detector
ACCIDENTAL_DETECTOR_AVAILABLE = False
try:
    from homr.detection.accidentals import AccidentalDetector, Detection
    ACCIDENTAL_DETECTOR_AVAILABLE = True
except ImportError:
    pass


def detect_accidentals_with_yolo(
    image: NDArray,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.3,
) -> List[Tuple[RotatedBoundingBox, str, float]]:
    """
    Detect accidentals in an image using YOLOv10.

    Args:
        image: Input image (BGR or grayscale)
        model_path: Path to YOLOv10 weights (None for default)
        confidence_threshold: Minimum confidence for detections

    Returns:
        List of (bounding_box, class_name, confidence) tuples
    """
    if not ACCIDENTAL_DETECTOR_AVAILABLE:
        print("Warning: AccidentalDetector not available. Install sahi and ultralytics.")
        return []

    # Convert to RGB if needed
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    try:
        detector = AccidentalDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        detections = detector.detect_in_image(img_rgb)

        results = []
        for det in detections:
            # Create a RotatedBoundingBox from the detection
            center = ((det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2)
            size = (det.x2 - det.x1, det.y2 - det.y1)

            # Create contours for the bounding box
            x1, y1 = int(det.x1), int(det.y1)
            x2, y2 = int(det.x2), int(det.y2)
            contours = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

            # RotatedBoundingBox expects ((center_x, center_y), (width, height), angle)
            rotated_rect = (center, size, 0.0)
            bbox = RotatedBoundingBox(rotated_rect, contours)

            results.append((bbox, det.class_name, det.confidence))

        return results
    except Exception as e:
        print(f"Error detecting accidentals: {e}")
        return []


def add_accidentals_to_staffs(
    staffs: List[Staff],
    accidental_detections: List[Tuple[RotatedBoundingBox, str, float]],
) -> List[Accidental]:
    """
    Assign detected accidentals to staffs and calculate their pitch positions.

    This uses the SAME staff positioning system as notes - finding the
    closest staff line at the accidental's X position and calculating
    the position in unit sizes.

    Args:
        staffs: List of detected Staff objects with grid points
        accidental_detections: List of (bbox, class_name, confidence) from YOLO

    Returns:
        List of Accidental objects with positions and pitch names
    """
    accidentals = []

    for bbox, class_name, confidence in accidental_detections:
        center = bbox.center

        # Find which staff this accidental belongs to
        best_staff = None
        best_distance = float('inf')

        for staff in staffs:
            # Check if the accidental is in this staff's zone
            if staff.is_on_staff_zone(bbox):
                distance = staff.y_distance_to(center)
                if distance < best_distance:
                    best_distance = distance
                    best_staff = staff

        if best_staff is None:
            # No staff found, try to find the closest one anyway
            for staff in staffs:
                distance = staff.y_distance_to(center)
                if distance < best_distance:
                    best_distance = distance
                    best_staff = staff

        if best_staff is not None:
            # Get the staff point at this X position
            point = best_staff.get_at(center[0])

            if point is not None:
                # Use the SAME positioning logic as notes!
                position = point.find_position_in_unit_sizes(bbox)

                # Convert class name to AccidentalType
                accidental_type = AccidentalType.from_class_name(class_name)

                # Create the Accidental object
                accidental = Accidental(
                    box=bbox,
                    position=position,
                    accidental_type=accidental_type,
                    confidence=confidence,
                )

                accidentals.append(accidental)

                # Also add to the staff's symbols
                best_staff.add_symbol(accidental)

    return accidentals


def detect_and_position_accidentals(
    image: NDArray,
    staffs: List[Staff],
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.3,
) -> List[Accidental]:
    """
    Full pipeline: detect accidentals with YOLO and assign positions using staff lines.

    Args:
        image: Input image (BGR or grayscale)
        staffs: List of detected Staff objects
        model_path: Path to YOLOv10 weights (None for default)
        confidence_threshold: Minimum confidence for detections

    Returns:
        List of Accidental objects with pitch names
    """
    # Step 1: Detect accidentals with YOLO
    detections = detect_accidentals_with_yolo(
        image,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )

    print(f"Detected {len(detections)} accidentals with YOLOv10")

    if len(detections) == 0:
        return []

    # Step 2: Assign to staffs and calculate positions
    accidentals = add_accidentals_to_staffs(staffs, detections)

    print(f"Positioned {len(accidentals)} accidentals on staffs")

    # Print summary
    type_counts = {}
    for acc in accidentals:
        type_name = acc.accidental_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print("Accidental breakdown:")
    for type_name, count in sorted(type_counts.items()):
        print(f"  {type_name}: {count}")

    return accidentals


def draw_accidentals_visualization(
    image: NDArray,
    accidentals: List[Accidental],
    staffs: List[Staff],
) -> NDArray:
    """
    Draw accidentals with their pitch names on an image.

    Args:
        image: Original image
        accidentals: List of Accidental objects with positions
        staffs: List of Staff objects (for drawing staff lines)

    Returns:
        Image with accidentals and pitch names drawn
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Reset accidental counter for consistent numbering
    Accidental.reset_accidental_counter()

    # Draw staff lines in gray
    for staff in staffs:
        staff.draw_onto_image(img, (128, 128, 128))

    # Sort accidentals by x position (left to right)
    sorted_accidentals = sorted(accidentals, key=lambda a: a.center[0])

    # Color map for different accidental types
    colors = {
        AccidentalType.SHARP: (0, 0, 255),       # Red
        AccidentalType.FLAT: (255, 0, 0),         # Blue
        AccidentalType.NATURAL: (0, 255, 0),      # Green
        AccidentalType.DOUBLE_SHARP: (0, 128, 255),  # Orange
        AccidentalType.DOUBLE_FLAT: (255, 0, 128),   # Purple
        AccidentalType.KEY_SHARP: (0, 0, 200),    # Dark Red
        AccidentalType.KEY_FLAT: (200, 0, 0),     # Dark Blue
        AccidentalType.KEY_NATURAL: (0, 200, 0),  # Dark Green
        AccidentalType.UNKNOWN: (128, 128, 128),  # Gray
    }

    # Draw each accidental
    for acc in sorted_accidentals:
        color = colors.get(acc.accidental_type, (128, 128, 128))
        acc.draw_onto_image(img, color)

    return img
