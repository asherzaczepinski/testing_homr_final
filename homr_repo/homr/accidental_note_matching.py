"""
Accidental-to-Note Matching Module.

This module matches detected accidentals to notes based on their pitch values,
using the detected letter names (e.g., F#5 affects all F notes in that measure).

Key features:
- Uses detected pitch values instead of raw staff line positions
- Handles octave equivalence (accidentals affect same note letter in all octaves)
- Tracks accidentals through measures (resets at bar lines)
- Naturals cancel previous accidentals
- Key signature accidentals persist through the whole piece
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from homr.model import Accidental, AccidentalType, Note, Staff, MultiStaff, BarLine
from homr.type_definitions import NDArray


@dataclass
class AccidentalNoteMatch:
    """Represents a match between an accidental and a note it affects."""
    accidental: Accidental
    note: Note
    accidental_pitch: str  # e.g., "F#5"
    note_pitch: str        # e.g., "F5"
    note_letter: str       # e.g., "F"


def extract_note_letter(pitch: str) -> str:
    """Extract just the letter from a pitch name (e.g., 'F#5' -> 'F', 'Bb3' -> 'B')."""
    if not pitch:
        return ""
    return pitch[0]


def extract_octave(pitch: str) -> Optional[int]:
    """Extract octave number from pitch (e.g., 'F#5' -> 5, 'Bb3' -> 3)."""
    if not pitch:
        return None
    # Find the last digit in the string
    for i in range(len(pitch) - 1, -1, -1):
        if pitch[i].isdigit():
            return int(pitch[i])
    return None


def get_accidental_modifier(acc_type: AccidentalType) -> str:
    """Get the modifier string for an accidental type."""
    if acc_type in [AccidentalType.SHARP, AccidentalType.KEY_SHARP]:
        return "sharp"
    elif acc_type in [AccidentalType.FLAT, AccidentalType.KEY_FLAT]:
        return "flat"
    elif acc_type in [AccidentalType.NATURAL, AccidentalType.KEY_NATURAL]:
        return "natural"
    elif acc_type == AccidentalType.DOUBLE_SHARP:
        return "double_sharp"
    elif acc_type == AccidentalType.DOUBLE_FLAT:
        return "double_flat"
    return "unknown"


def is_key_signature_accidental(acc_type: AccidentalType) -> bool:
    """Check if this is a key signature accidental (persists through piece)."""
    return acc_type in [AccidentalType.KEY_SHARP, AccidentalType.KEY_FLAT, AccidentalType.KEY_NATURAL]


def match_accidentals_to_notes(
    staffs: list[Staff],
    accidentals: list[Accidental],
    notes: list[Note],
    clef: str = "treble"
) -> list[AccidentalNoteMatch]:
    """
    Match accidentals to notes they affect based on pitch values.

    Rules:
    1. Accidentals only affect notes AFTER them (to the right) in the measure
    2. Accidentals affect the same note letter in ALL octaves
    3. Regular accidentals reset at bar lines
    4. Key signature accidentals persist through the piece
    5. Naturals cancel previous accidentals for that note letter

    Args:
        staffs: List of Staff objects with bar lines
        accidentals: List of detected Accidental objects
        notes: List of detected Note objects
        clef: "treble" or "bass" for pitch calculation

    Returns:
        List of AccidentalNoteMatch objects
    """
    matches = []

    # Process each staff independently
    for staff in staffs:
        # Get bar line X positions for this staff
        bar_lines = staff.get_bar_lines()
        bar_x_positions = sorted([bl.center[0] for bl in bar_lines])

        # Filter accidentals and notes that belong to this staff
        staff_accidentals = [
            acc for acc in accidentals
            if staff.is_on_staff_zone(acc.box)
        ]
        staff_notes = [
            note for note in notes
            if staff.is_on_staff_zone(note.box)
        ]

        # Sort by X position (left to right)
        staff_accidentals.sort(key=lambda a: a.center[0])
        staff_notes.sort(key=lambda n: n.center[0])

        # Track active accidentals: note_letter -> (AccidentalType, start_x)
        # Key signature accidentals are tracked separately
        key_signature_accidentals: dict[str, AccidentalType] = {}
        measure_accidentals: dict[str, tuple[AccidentalType, float]] = {}

        # Current bar line index
        current_bar_idx = 0

        # Process accidentals from left to right
        for acc in staff_accidentals:
            acc_x = acc.center[0]
            acc_pitch = acc.get_pitch_name(clef)
            note_letter = extract_note_letter(acc_pitch)

            # Check if we crossed a bar line (reset measure accidentals)
            while current_bar_idx < len(bar_x_positions) and acc_x > bar_x_positions[current_bar_idx]:
                measure_accidentals.clear()
                current_bar_idx += 1

            # Handle key signature vs regular accidentals
            if is_key_signature_accidental(acc.accidental_type):
                if acc.accidental_type == AccidentalType.KEY_NATURAL:
                    # Key natural cancels key signature for this note
                    if note_letter in key_signature_accidentals:
                        del key_signature_accidentals[note_letter]
                else:
                    key_signature_accidentals[note_letter] = acc.accidental_type
            else:
                if acc.accidental_type == AccidentalType.NATURAL:
                    # Natural cancels both key signature and measure accidentals for this note
                    if note_letter in measure_accidentals:
                        del measure_accidentals[note_letter]
                else:
                    measure_accidentals[note_letter] = (acc.accidental_type, acc_x)

        # Reset for note matching pass
        measure_accidentals.clear()
        current_bar_idx = 0

        # Track accidentals as we scan notes
        for acc in staff_accidentals:
            acc_x = acc.center[0]
            acc_pitch = acc.get_pitch_name(clef)
            note_letter = extract_note_letter(acc_pitch)

            # Check if we crossed a bar line
            while current_bar_idx < len(bar_x_positions) and acc_x > bar_x_positions[current_bar_idx]:
                measure_accidentals.clear()
                current_bar_idx += 1

            # Update tracking
            if is_key_signature_accidental(acc.accidental_type):
                if acc.accidental_type == AccidentalType.KEY_NATURAL:
                    if note_letter in key_signature_accidentals:
                        del key_signature_accidentals[note_letter]
                else:
                    key_signature_accidentals[note_letter] = acc.accidental_type
            else:
                if acc.accidental_type == AccidentalType.NATURAL:
                    if note_letter in measure_accidentals:
                        del measure_accidentals[note_letter]
                else:
                    measure_accidentals[note_letter] = (acc.accidental_type, acc_x)

            # Find notes affected by this accidental
            for note in staff_notes:
                note_x = note.center[0]
                note_pitch = note.get_pitch_name(clef)
                note_letter_from_note = extract_note_letter(note_pitch)

                # Note must be AFTER the accidental
                if note_x <= acc_x:
                    continue

                # For key signature accidentals, affect all matching notes
                # For measure accidentals, check if within same measure
                if is_key_signature_accidental(acc.accidental_type):
                    # Key signature affects all matching notes
                    if note_letter_from_note == note_letter:
                        match = AccidentalNoteMatch(
                            accidental=acc,
                            note=note,
                            accidental_pitch=acc_pitch,
                            note_pitch=note_pitch,
                            note_letter=note_letter
                        )
                        matches.append(match)
                else:
                    # Measure accidental - check if in same measure
                    # Find bar lines between accidental and note
                    bars_between = [bx for bx in bar_x_positions if acc_x < bx < note_x]

                    if len(bars_between) == 0 and note_letter_from_note == note_letter:
                        # Same measure and same note letter
                        match = AccidentalNoteMatch(
                            accidental=acc,
                            note=note,
                            accidental_pitch=acc_pitch,
                            note_pitch=note_pitch,
                            note_letter=note_letter
                        )
                        matches.append(match)

    return matches


def draw_accidental_note_connections(
    image: NDArray,
    matches: list[AccidentalNoteMatch],
    staffs: list[Staff],
) -> NDArray:
    """
    Draw visualization showing which accidentals affect which notes.

    - Draws lines connecting accidentals to their affected notes
    - Circles affected notes with color based on accidental type
    - Labels show the pitch relationship

    Args:
        image: Original image to draw on
        matches: List of AccidentalNoteMatch objects
        staffs: List of Staff objects (for drawing staff lines)

    Returns:
        Image with connections drawn
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw staff lines in light gray
    for staff in staffs:
        staff.draw_onto_image(img, (180, 180, 180))

    # Color map for different accidental types
    colors = {
        AccidentalType.SHARP: (0, 0, 255),       # Red
        AccidentalType.FLAT: (255, 165, 0),      # Orange
        AccidentalType.NATURAL: (0, 255, 0),     # Green
        AccidentalType.DOUBLE_SHARP: (0, 0, 200),
        AccidentalType.DOUBLE_FLAT: (200, 100, 0),
        AccidentalType.KEY_SHARP: (100, 100, 255),  # Light red
        AccidentalType.KEY_FLAT: (255, 200, 100),   # Light orange
        AccidentalType.KEY_NATURAL: (100, 255, 100), # Light green
    }

    # Group matches by accidental for cleaner visualization
    acc_to_notes: dict[int, list[AccidentalNoteMatch]] = {}
    for match in matches:
        acc_id = id(match.accidental)
        if acc_id not in acc_to_notes:
            acc_to_notes[acc_id] = []
        acc_to_notes[acc_id].append(match)

    # Draw each accidental and its connected notes
    for acc_id, acc_matches in acc_to_notes.items():
        if not acc_matches:
            continue

        acc = acc_matches[0].accidental
        color = colors.get(acc.accidental_type, (128, 128, 128))

        # Draw accidental box
        acc.box.draw_onto_image(img, color)

        # Draw label for accidental
        acc_label = f"{acc.accidental_type.get_symbol()}{acc_matches[0].note_letter}"
        cv2.putText(
            img,
            acc_label,
            (int(acc.center[0]) - 10, int(acc.center[1]) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

        # Draw connections and circles for each affected note
        for match in acc_matches:
            note = match.note

            # Draw line from accidental to note
            cv2.line(
                img,
                (int(acc.center[0]), int(acc.center[1])),
                (int(note.center[0]), int(note.center[1])),
                color,
                1,
                cv2.LINE_AA
            )

            # Circle the affected note
            center = (int(note.center[0]), int(note.center[1]))
            # Get note size for ellipse
            if hasattr(note.box, 'size'):
                axes = (int(note.box.size[0] / 2 + 8), int(note.box.size[1] / 2 + 8))
            else:
                axes = (15, 10)

            cv2.ellipse(img, center, axes, 0, 0, 360, color, 2)

            # Small label showing the note pitch
            cv2.putText(
                img,
                match.note_pitch,
                (center[0] + axes[0] + 2, center[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA
            )

    return img


def create_accidental_effects_summary(
    matches: list[AccidentalNoteMatch]
) -> str:
    """
    Create a text summary of accidental effects.

    Args:
        matches: List of AccidentalNoteMatch objects

    Returns:
        Summary string
    """
    if not matches:
        return "No accidental-note matches found."

    lines = ["Accidental Effects Summary:", "=" * 40]

    # Group by accidental
    acc_to_notes: dict[str, list[str]] = {}
    for match in matches:
        acc_key = f"{match.accidental_pitch} ({match.accidental.accidental_type.value})"
        if acc_key not in acc_to_notes:
            acc_to_notes[acc_key] = []
        acc_to_notes[acc_key].append(match.note_pitch)

    for acc_key, note_pitches in acc_to_notes.items():
        lines.append(f"\n{acc_key}:")
        lines.append(f"  Affects {len(note_pitches)} notes: {', '.join(note_pitches[:10])}")
        if len(note_pitches) > 10:
            lines.append(f"  ... and {len(note_pitches) - 10} more")

    return "\n".join(lines)
