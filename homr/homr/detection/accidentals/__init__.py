"""
Accidental detection module.
"""

from homr.detection.accidentals.accidental_detector import (
    AccidentalDetector,
    ACCIDENTAL_CLASSES,
)
from homr.detection.accidentals.notehead_detector import (
    NoteHeadDetector,
    Detection,
    draw_detections,
    deduplicate_detections,
)

__all__ = [
    "AccidentalDetector",
    "ACCIDENTAL_CLASSES",
    "NoteHeadDetector",
    "Detection",
    "draw_detections",
    "deduplicate_detections",
]
