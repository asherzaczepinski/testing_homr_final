"""
Accidental detection with staff positioning integration.

This module combines TWO YOLOv10 accidental detection models:
1. Custom accidental detector (homr.detection.accidentals)
2. DeepScoresV2 detector (scanner_deepscores) - has precedence on overlaps

Both models work together to detect sharps, flats, and naturals.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

from homr.model import (
    Accidental,
    AccidentalType,
    Staff,
    note_names,
)
from homr.bounding_boxes import RotatedBoundingBox, BoundingBox
from homr.type_definitions import NDArray

# Try to import the custom YOLOv10 detector
ACCIDENTAL_DETECTOR_AVAILABLE = False
try:
    from homr.detection.accidentals import AccidentalDetector, Detection
    ACCIDENTAL_DETECTOR_AVAILABLE = True
except ImportError:
    pass

# Try to import the DeepScoresV2 detector
DEEPSCORES_DETECTOR_AVAILABLE = False
try:
    from homr.scanner_deepscores.detection import detect_everything, filter_predictions
    DEEPSCORES_DETECTOR_AVAILABLE = True
except ImportError:
    pass


def detect_accidentals_with_yolo(
    image: NDArray,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.3,
    unit_size: Optional[float] = None,
    max_width_units: float = 2.0,
    max_height_units: float = 3.0,
) -> List[Tuple[RotatedBoundingBox, str, float]]:
    """
    Detect accidentals in an image using YOLOv10.

    Size filtering is relative to staff line spacing (unit_size):
    - Width: max 2 line spacings (accidentals are narrow)
    - Height: max 3 line spacings (accidentals span about 3 lines)

    Args:
        image: Input image (BGR or grayscale)
        model_path: Path to YOLOv10 weights (None for default)
        confidence_threshold: Minimum confidence for detections
        unit_size: Distance between staff lines in pixels (if None, uses fallback pixel values)
        max_width_units: Maximum width as multiple of unit_size (default: 2 lines)
        max_height_units: Maximum height as multiple of unit_size (default: 3 lines)

    Returns:
        List of (bounding_box, class_name, confidence) tuples
    """
    if not ACCIDENTAL_DETECTOR_AVAILABLE:
        print("Warning: AccidentalDetector not available. Install sahi and ultralytics.")
        return []

    # Calculate size limits based on unit_size (relative to staff line spacing)
    if unit_size is not None and unit_size > 0:
        # Minimum size: about 0.3 line spacings
        min_width = int(unit_size * 0.3)
        max_width = int(unit_size * max_width_units)
        min_height = int(unit_size * 0.8)
        max_height = int(unit_size * max_height_units)
        print(f"Accidental size filtering (unit_size={unit_size:.1f}px): width {min_width}-{max_width}px, height {min_height}-{max_height}px")
    else:
        # Fallback to reasonable pixel values if no unit_size provided
        min_width, max_width = 5, 50
        min_height, max_height = 15, 70
        print(f"Accidental size filtering (fallback): width {min_width}-{max_width}px, height {min_height}-{max_height}px")

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
        filtered_count = 0
        for det in detections:
            # Calculate size
            width = det.x2 - det.x1
            height = det.y2 - det.y1

            # Filter by size - skip if too big or too small
            if width < min_width or width > max_width:
                filtered_count += 1
                continue
            if height < min_height or height > max_height:
                filtered_count += 1
                continue

            # Create a RotatedBoundingBox from the detection
            center = ((det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2)
            size = (width, height)

            # Create contours for the bounding box
            x1, y1 = int(det.x1), int(det.y1)
            x2, y2 = int(det.x2), int(det.y2)
            contours = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

            # RotatedBoundingBox expects ((center_x, center_y), (width, height), angle)
            rotated_rect = (center, size, 0.0)
            bbox = RotatedBoundingBox(rotated_rect, contours)

            results.append((bbox, det.class_name, det.confidence))

        if filtered_count > 0:
            print(f"Filtered out {filtered_count} detections by size (relative to staff lines)")

        return results
    except Exception as e:
        print(f"Error detecting accidentals: {e}")
        return []


def detect_accidentals_with_deepscores(
    image: NDArray,
    confidence_threshold: float = 0.5,
    unit_size: Optional[float] = None,
    max_width_units: float = 2.0,
    max_height_units: float = 3.0,
) -> List[Tuple[RotatedBoundingBox, str, float]]:
    """
    Detect accidentals using the DeepScoresV2 model.

    This model was trained on DeepScoresV2 dataset with 135 classes including:
    - keyFlat, keySharp, keyNatural (key signature accidentals)
    - accidentalFlat, accidentalSharp, accidentalNatural (in-measure accidentals)
    - accidentalDoubleSharp, accidentalDoubleFlat

    Args:
        image: Input image (BGR or grayscale)
        confidence_threshold: Minimum confidence for detections
        unit_size: Distance between staff lines in pixels
        max_width_units: Maximum width as multiple of unit_size
        max_height_units: Maximum height as multiple of unit_size

    Returns:
        List of (bounding_box, class_name, confidence) tuples
    """
    if not DEEPSCORES_DETECTOR_AVAILABLE:
        print("Warning: DeepScoresV2 detector not available.")
        return []

    # Calculate size limits
    if unit_size is not None and unit_size > 0:
        min_width = int(unit_size * 0.3)
        max_width = int(unit_size * max_width_units)
        min_height = int(unit_size * 0.8)
        max_height = int(unit_size * max_height_units)
    else:
        min_width, max_width = 5, 50
        min_height, max_height = 15, 70

    # Convert to RGB if needed
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image

    try:
        # Run detection
        result = detect_everything(img_rgb)

        # Filter for accidentals only
        accidental_categories = {
            'keyFlat', 'keySharp', 'keyNatural',
            'accidentalFlat', 'accidentalSharp', 'accidentalNatural',
            'accidentalDoubleSharp', 'accidentalDoubleFlat'
        }

        # Map DeepScores class names to our class names
        class_name_map = {
            'keyFlat': 'key_flat',
            'keySharp': 'key_sharp',
            'keyNatural': 'key_natural',
            'accidentalFlat': 'flat',
            'accidentalSharp': 'sharp',
            'accidentalNatural': 'natural',
            'accidentalDoubleSharp': 'double_sharp',
            'accidentalDoubleFlat': 'double_flat',
        }

        results = []
        filtered_count = 0

        for pred in result.object_prediction_list:
            if pred.category.name not in accidental_categories:
                continue

            # Get bounding box
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            width = x2 - x1
            height = y2 - y1

            # Filter by size
            if width < min_width or width > max_width:
                filtered_count += 1
                continue
            if height < min_height or height > max_height:
                filtered_count += 1
                continue

            # Create RotatedBoundingBox
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (width, height)
            contours = np.array([
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)]
            ], dtype=np.int32)

            rotated_rect = (center, size, 0.0)
            bbox = RotatedBoundingBox(rotated_rect, contours)

            # Map class name
            class_name = class_name_map.get(pred.category.name, pred.category.name)
            confidence = pred.score.value

            results.append((bbox, class_name, confidence))

        if filtered_count > 0:
            print(f"DeepScores: Filtered out {filtered_count} detections by size")

        return results

    except Exception as e:
        print(f"Error in DeepScores detection: {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_iou(box1: RotatedBoundingBox, box2: RotatedBoundingBox) -> float:
    """Calculate Intersection over Union between two boxes."""
    # Get bounding rectangles
    x1_min, y1_min = box1.center[0] - box1.size[0]/2, box1.center[1] - box1.size[1]/2
    x1_max, y1_max = box1.center[0] + box1.size[0]/2, box1.center[1] + box1.size[1]/2

    x2_min, y2_min = box2.center[0] - box2.size[0]/2, box2.center[1] - box2.size[1]/2
    x2_max, y2_max = box2.center[0] + box2.size[0]/2, box2.center[1] + box2.size[1]/2

    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)

    # Calculate union
    area1 = box1.size[0] * box1.size[1]
    area2 = box2.size[0] * box2.size[1]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def merge_detections(
    primary_detections: List[Tuple[RotatedBoundingBox, str, float]],
    secondary_detections: List[Tuple[RotatedBoundingBox, str, float]],
    iou_threshold: float = 0.3,
) -> List[Tuple[RotatedBoundingBox, str, float]]:
    """
    Merge detections from two models.

    Primary detections (DeepScores) have precedence:
    - All primary detections are kept
    - Secondary detections are only added if they don't overlap with primary

    Args:
        primary_detections: Detections from primary model (DeepScores) - has precedence
        secondary_detections: Detections from secondary model (custom YOLO)
        iou_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        Merged list of detections
    """
    merged = list(primary_detections)  # Keep all primary detections

    added_count = 0
    for sec_bbox, sec_class, sec_conf in secondary_detections:
        # Check if this detection overlaps with any primary detection
        overlaps = False
        for pri_bbox, _, _ in primary_detections:
            iou = calculate_iou(sec_bbox, pri_bbox)
            if iou > iou_threshold:
                overlaps = True
                break

        if not overlaps:
            # This is an isolated detection from secondary model - add it
            merged.append((sec_bbox, sec_class, sec_conf))
            added_count += 1

    if added_count > 0:
        print(f"Added {added_count} isolated detections from secondary model")

    return merged


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
    Full pipeline: detect accidentals with TWO YOLO models and assign positions.

    Uses TWO detection models that work together:
    1. DeepScoresV2 model (primary) - has precedence on overlaps
    2. Custom accidental model (secondary) - fills in isolated detections

    Size filtering is relative to staff line spacing:
    - Width: max 2 line spacings
    - Height: max 3 line spacings

    Args:
        image: Input image (BGR or grayscale)
        staffs: List of detected Staff objects
        model_path: Path to custom YOLOv10 weights (None for default)
        confidence_threshold: Minimum confidence for detections

    Returns:
        List of Accidental objects with pitch names
    """
    # Calculate average unit size from all staffs (distance between staff lines)
    unit_size = None
    if staffs:
        unit_sizes = [staff.average_unit_size for staff in staffs if staff.average_unit_size > 0]
        if unit_sizes:
            unit_size = float(np.mean(unit_sizes))
            print(f"Staff line spacing (unit_size): {unit_size:.1f}px")

    # Step 1a: Detect with DeepScoresV2 model (PRIMARY - has precedence)
    print("Running DeepScoresV2 accidental detection...")
    deepscores_detections = detect_accidentals_with_deepscores(
        image,
        confidence_threshold=0.5,  # DeepScores uses higher threshold
        unit_size=unit_size,
        max_width_units=2.0,
        max_height_units=3.0,
    )
    print(f"DeepScores detected {len(deepscores_detections)} accidentals")

    # Step 1b: Detect with custom YOLO model (SECONDARY - fills gaps)
    print("Running custom YOLO accidental detection...")
    yolo_detections = detect_accidentals_with_yolo(
        image,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        unit_size=unit_size,
        max_width_units=2.0,
        max_height_units=3.0,
    )
    print(f"Custom YOLO detected {len(yolo_detections)} accidentals")

    # Step 2: Merge detections (DeepScores has precedence, YOLO adds isolated ones)
    merged_detections = merge_detections(
        primary_detections=deepscores_detections,
        secondary_detections=yolo_detections,
        iou_threshold=0.3,
    )
    print(f"Total merged detections: {len(merged_detections)}")

    if len(merged_detections) == 0:
        return []

    # Step 3: Assign to staffs and calculate positions
    accidentals = add_accidentals_to_staffs(staffs, merged_detections)

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
