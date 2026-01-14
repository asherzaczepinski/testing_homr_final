"""
DeepScore-based note head detector using YOLOv10 with SAHI sliced inference.
Filters detections to only include note heads (not sharps, flats, etc.)
"""

import os
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

# SAHI imports
try:
    from sahi.predict import get_sliced_prediction
    from sahi.prediction import ObjectPrediction
    from sahi.models.base import DetectionModel
    from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("Warning: SAHI not installed. Run: pip install sahi")

# Ultralytics imports
try:
    from ultralytics import YOLOv10
    YOLO_AVAILABLE = True
except ImportError:
    try:
        from ultralytics import YOLO
        YOLOv10 = YOLO  # Fallback to regular YOLO
        YOLO_AVAILABLE = True
    except ImportError:
        YOLO_AVAILABLE = False
        print("Warning: ultralytics not installed. Run: pip install ultralytics")


# Note head class names from DeepScores dataset (indices 24-39)
NOTEHEAD_CLASSES = {
    'noteheadBlackOnLine',
    'noteheadBlackOnLineSmall',
    'noteheadBlackInSpace',
    'noteheadBlackInSpaceSmall',
    'noteheadHalfOnLine',
    'noteheadHalfOnLineSmall',
    'noteheadHalfInSpace',
    'noteheadHalfInSpaceSmall',
    'noteheadWholeOnLine',
    'noteheadWholeOnLineSmall',
    'noteheadWholeInSpace',
    'noteheadWholeInSpaceSmall',
    'noteheadDoubleWholeOnLine',
    'noteheadDoubleWholeOnLineSmall',
    'noteheadDoubleWholeInSpace',
    'noteheadDoubleWholeInSpaceSmall',
}

# Classes to exclude (accidentals, etc.)
EXCLUDED_CLASSES = {
    'accidentalFlat',
    'accidentalFlatSmall',
    'accidentalNatural',
    'accidentalNaturalSmall',
    'accidentalSharp',
    'accidentalSharpSmall',
    'accidentalDoubleSharp',
    'accidentalDoubleFlat',
    'keyFlat',
    'keyNatural',
    'keySharp',
}


@dataclass
class Detection:
    """Represents a single detected note head."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    class_id: int

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: 'Detection') -> float:
        """Calculate Intersection over Union with another detection."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class Yolov10DetectionModel(DetectionModel):
    """YOLOv10 detection model wrapper for SAHI."""

    def check_dependencies(self):
        pass

    def load_model(self):
        """Load the YOLOv10 model."""
        try:
            model = YOLOv10(self.model_path)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError(f"Failed to load model from {self.model_path}: {e}")

    def set_model(self, model):
        """Set the model and category mapping."""
        self.model = model
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """Run inference on the image."""
        if self.model is None:
            raise ValueError("Model not loaded")

        kwargs = {
            "verbose": False,
            "conf": self.confidence_threshold,
            "device": self.device
        }

        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size

        # YOLOv10 expects BGR format
        prediction_result = self.model(image[:, :, ::-1], **kwargs)
        prediction_result = [result.boxes.data for result in prediction_result]

        self._original_predictions = prediction_result
        self._original_shape = image.shape

    @property
    def category_names(self):
        return self.model.names.values()

    @property
    def num_categories(self):
        return len(self.model.names)

    @property
    def has_mask(self):
        return False

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """Convert raw predictions to ObjectPrediction list."""
        original_predictions = self._original_predictions

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            for prediction in image_predictions.cpu().detach().numpy():
                x1, y1, x2, y2 = prediction[:4]
                bbox = [max(0, x1), max(0, y1), max(0, x2), max(0, y2)]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


class NoteHeadDetector:
    """
    Detector for note heads in sheet music images using DeepScores YOLOv10 model.
    Filters to only return note heads, excluding accidentals.
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        device: str = None,
        notehead_classes: Set[str] = None
    ):
        """
        Initialize the note head detector.

        Args:
            model_path: Path to the YOLOv10 weights file (.pt)
            confidence_threshold: Minimum confidence for detections
            device: 'cuda' or 'cpu' (auto-detected if None)
            notehead_classes: Set of class names to detect (defaults to NOTEHEAD_CLASSES)
        """
        if not SAHI_AVAILABLE or not YOLO_AVAILABLE:
            raise ImportError("Required packages not installed. Run: pip install sahi ultralytics")

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # Find model path
        if model_path is None:
            # Try common locations (internal homr location first, then fallbacks)
            possible_paths = [
                # New internal location (relative to this file: homr/detection/accidentals/)
                os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'accidentals', 'best.pt'),
                # Old Orchestra-AI-2 locations (for development/fallback)
                os.path.join(os.path.dirname(__file__), 'weights', 'best.pt'),
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Orchestra-AI-2', 'weights', 'best.pt'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                raise FileNotFoundError(
                    "Could not find model weights. Please provide model_path or ensure "
                    "best.pt exists at homr/models/accidentals/best.pt"
                )

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.notehead_classes = notehead_classes or NOTEHEAD_CLASSES

        # Initialize the detection model
        print(f"Loading model from {model_path} on {device}...")
        self.detection_model = Yolov10DetectionModel(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        print("Model loaded successfully.")

    def detect_in_image(
        self,
        image: np.ndarray,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.5,
    ) -> List[Detection]:
        """
        Detect note heads in a full image using sliced inference.

        Args:
            image: Input image (RGB numpy array)
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Vertical overlap between slices
            overlap_width_ratio: Horizontal overlap between slices

        Returns:
            List of Detection objects for note heads only
        """
        # Run sliced prediction
        result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_type="NMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.1,
        )

        # Filter to only note heads
        detections = []
        for pred in result.object_prediction_list:
            if pred.category.name in self.notehead_classes:
                bbox = pred.bbox.to_xyxy()
                detections.append(Detection(
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    confidence=pred.score.value,
                    class_name=pred.category.name,
                    class_id=pred.category.id,
                ))

        return detections

    def detect_in_measure(
        self,
        measure_image: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Detection]:
        """
        Detect note heads in a single measure image.

        Args:
            measure_image: Cropped measure image (RGB or grayscale)
            offset_x: X offset to add to detection coordinates (for mapping back to original)
            offset_y: Y offset to add to detection coordinates

        Returns:
            List of Detection objects with coordinates relative to original image
        """
        # Convert grayscale to RGB if needed
        if len(measure_image.shape) == 2:
            measure_image = cv2.cvtColor(measure_image, cv2.COLOR_GRAY2RGB)
        elif measure_image.shape[2] == 4:
            measure_image = cv2.cvtColor(measure_image, cv2.COLOR_BGRA2RGB)
        elif measure_image.shape[2] == 3:
            # Check if it's BGR and convert
            measure_image = cv2.cvtColor(measure_image, cv2.COLOR_BGR2RGB)

        # For small measures, don't use slicing
        h, w = measure_image.shape[:2]
        if h <= 640 and w <= 640:
            # Direct inference without slicing
            self.detection_model.perform_inference(measure_image)
            self.detection_model._create_object_prediction_list_from_original_predictions()
            predictions = self.detection_model._object_prediction_list_per_image[0]
        else:
            # Use sliced inference
            result = get_sliced_prediction(
                measure_image,
                self.detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.5,
                postprocess_type="NMM",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.1,
            )
            predictions = result.object_prediction_list

        # Filter to note heads and apply offset
        detections = []
        for pred in predictions:
            if pred.category.name in self.notehead_classes:
                bbox = pred.bbox.to_xyxy()
                detections.append(Detection(
                    x1=bbox[0] + offset_x,
                    y1=bbox[1] + offset_y,
                    x2=bbox[2] + offset_x,
                    y2=bbox[3] + offset_y,
                    confidence=pred.score.value,
                    class_name=pred.category.name,
                    class_id=pred.category.id,
                ))

        return detections


def deduplicate_detections(
    all_detections: List[Detection],
    iou_threshold: float = 0.5
) -> List[Detection]:
    """
    Remove duplicate detections from overlapping measures.
    When two detections overlap (same symbol detected from multiple measures),
    keep only the one with higher confidence.

    Args:
        all_detections: List of all detections from all measures
        iou_threshold: IoU threshold above which detections are considered duplicates

    Returns:
        Deduplicated list of detections
    """
    if not all_detections:
        return []

    # Sort by confidence (highest first)
    sorted_dets = sorted(all_detections, key=lambda d: d.confidence, reverse=True)

    # Greedy NMS-like deduplication
    keep = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)

        # Remove detections that overlap too much with the best one
        sorted_dets = [d for d in sorted_dets if d.iou(best) < iou_threshold]

    return keep


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw detection boxes on an image.

    Args:
        image: Input image (will be copied)
        detections: List of detections to draw
        color: BGR color for boxes
        thickness: Line thickness
        show_labels: Whether to show class names
        show_confidence: Whether to show confidence scores

    Returns:
        Image with drawn detections
    """
    result = image.copy()
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if show_labels or show_confidence:
            label_parts = []
            if show_labels:
                # Shorten class name
                short_name = det.class_name.replace('notehead', '').replace('OnLine', 'L').replace('InSpace', 'S')
                label_parts.append(short_name)
            if show_confidence:
                label_parts.append(f"{det.confidence:.2f}")

            label = " ".join(label_parts)

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(result, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return result


# ============================================================================
# Test function for single image testing
# ============================================================================

def test_single_image(
    image_path: str,
    output_path: str = None,
    model_path: str = None,
    confidence_threshold: float = 0.5,
):
    """
    Test the note head detector on a single image.

    Args:
        image_path: Path to input image
        output_path: Path for output image (defaults to input_detected.png)
        model_path: Path to model weights
        confidence_threshold: Detection confidence threshold
    """
    print(f"\n{'='*60}")
    print("Note Head Detection Test")
    print(f"{'='*60}")

    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Initialize detector
    print("\nInitializing detector...")
    detector = NoteHeadDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )

    # Run detection
    print("\nRunning detection...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_in_image(image_rgb)

    print(f"\nFound {len(detections)} note heads:")

    # Group by class
    class_counts = {}
    for det in detections:
        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")

    # Draw detections
    result = draw_detections(image, detections, color=(0, 255, 0), thickness=2)

    # Save result
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"

    cv2.imwrite(output_path, result)
    print(f"\nSaved result to: {output_path}")

    return detections


# Test code removed - this module is now integrated into homr
# For testing, use the AccidentalDetector class from accidental_detector.py
