"""
Accidental detection module for the homr OMR system.

This module provides high-accuracy detection of musical accidentals
using a YOLOv10 model trained on the DeepScores dataset.

Extracted from Orchestra-AI-2 project.
"""

import cv2
import numpy as np
from homr.detection.accidentals.notehead_detector import NoteHeadDetector, draw_detections

# Accidental classes from DeepScores
ACCIDENTAL_CLASSES = {
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


class AccidentalDetector(NoteHeadDetector):
    """
    Detector specifically for musical accidentals.

    Uses YOLOv10 model trained on DeepScores dataset to detect sharps, flats,
    naturals, double sharps, and double flats in sheet music images.

    Args:
        model_path: Path to model weights file. If None, uses default location.
        confidence_threshold: Minimum confidence score for detections (0.0-1.0).
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect).

    Example:
        detector = AccidentalDetector(confidence_threshold=0.3)
        detections = detector.detect_in_image(image_rgb)
    """

    def __init__(self, model_path=None, confidence_threshold=0.3, device=None):
        # Initialize with accidental classes instead of notehead classes
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            notehead_classes=ACCIDENTAL_CLASSES
        )
