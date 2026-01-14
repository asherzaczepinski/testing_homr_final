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
    4. Key signature accidentals persist through the piece (but can be cancelled by naturals)
    5. Naturals CANCEL previous accidentals - notes after a natural are NOT affected
       until another sharp/flat is seen

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

        # Combine accidentals and notes into a single timeline, sorted by X
        # Each item is (x_position, 'acc' or 'note', object)
        timeline = []
        for acc in staff_accidentals:
            timeline.append((acc.center[0], 'acc', acc))
        for note in staff_notes:
            timeline.append((note.center[0], 'note', note))
        timeline.sort(key=lambda x: x[0])

        # Track active effects per note letter
        # key_effects: note_letter -> (AccidentalType, Accidental) - persists across measures
        # measure_effects: note_letter -> (AccidentalType, Accidental) - resets at bar lines
        # cancelled_letters: set of note letters that have been cancelled by naturals this measure
        key_effects: dict[str, tuple[AccidentalType, Accidental]] = {}
        measure_effects: dict[str, tuple[AccidentalType, Accidental]] = {}
        cancelled_in_measure: set[str] = set()  # Letters cancelled by naturals this measure

        current_bar_idx = 0

        for x_pos, item_type, obj in timeline:
            # Check if we crossed a bar line (reset measure accidentals and cancellations)
            while current_bar_idx < len(bar_x_positions) and x_pos > bar_x_positions[current_bar_idx]:
                measure_effects.clear()
                cancelled_in_measure.clear()
                current_bar_idx += 1

            if item_type == 'acc':
                acc = obj
                acc_pitch = acc.get_pitch_name(clef)
                note_letter = extract_note_letter(acc_pitch)

                if is_key_signature_accidental(acc.accidental_type):
                    if acc.accidental_type == AccidentalType.KEY_NATURAL:
                        # Key natural cancels key signature effect
                        if note_letter in key_effects:
                            del key_effects[note_letter]
                    else:
                        # Key sharp/flat - add to key effects
                        key_effects[note_letter] = (acc.accidental_type, acc)
                        # Also remove from cancelled set since we have a new accidental
                        cancelled_in_measure.discard(note_letter)
                else:
                    if acc.accidental_type == AccidentalType.NATURAL:
                        # Natural cancels effects - add to cancelled set
                        if note_letter in measure_effects:
                            del measure_effects[note_letter]
                        cancelled_in_measure.add(note_letter)
                    else:
                        # Sharp/flat - add to measure effects
                        measure_effects[note_letter] = (acc.accidental_type, acc)
                        # Remove from cancelled set since we have a new accidental
                        cancelled_in_measure.discard(note_letter)

            elif item_type == 'note':
                note = obj
                note_pitch = note.get_pitch_name(clef)
                note_letter = extract_note_letter(note_pitch)

                # Skip if this note letter has been cancelled by a natural this measure
                if note_letter in cancelled_in_measure:
                    continue

                # Check if this note is affected by an active accidental
                active_acc = None
                active_type = None

                # Measure effects take precedence over key effects
                if note_letter in measure_effects:
                    active_type, active_acc = measure_effects[note_letter]
                elif note_letter in key_effects:
                    active_type, active_acc = key_effects[note_letter]

                if active_acc is not None and active_type is not None:
                    # This note is affected by an accidental
                    match = AccidentalNoteMatch(
                        accidental=active_acc,
                        note=note,
                        accidental_pitch=active_acc.get_pitch_name(clef),
                        note_pitch=note_pitch,
                        note_letter=note_letter
                    )
                    matches.append(match)

    return matches


def get_accidental_color(acc_type: AccidentalType) -> tuple[int, int, int]:
    """Get the color for an accidental type. Red for sharps, light blue for flats."""
    # Red for sharps, light blue for flats
    # Naturals don't get highlighted (return None handled by caller)
    colors = {
        AccidentalType.SHARP: (0, 0, 255),         # Red (BGR)
        AccidentalType.FLAT: (255, 191, 0),        # Light blue (BGR)
        AccidentalType.DOUBLE_SHARP: (0, 0, 255),  # Red
        AccidentalType.DOUBLE_FLAT: (255, 191, 0), # Light blue
        AccidentalType.KEY_SHARP: (0, 0, 255),     # Red
        AccidentalType.KEY_FLAT: (255, 191, 0),    # Light blue
    }
    return colors.get(acc_type, None)


def draw_accidental_note_connections(
    image: NDArray,
    matches: list[AccidentalNoteMatch],
    staffs: list[Staff],
) -> NDArray:
    """
    Draw visualization showing which notes are affected by accidentals.

    - Draws ellipses around affected notes (5px buffer around notehead)
    - Red for sharps, light blue for flats
    - Naturals are NOT highlighted
    - No connecting lines - just the ellipses around affected notes

    Args:
        image: Original image to draw on
        matches: List of AccidentalNoteMatch objects
        staffs: List of Staff objects (for drawing staff lines)

    Returns:
        Image with affected notes highlighted
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Track which notes we've already drawn to avoid duplicates
    drawn_notes: set[int] = set()

    # Draw ellipses around each affected note
    for match in matches:
        note = match.note
        note_id = id(note)

        # Skip if we've already drawn this note
        if note_id in drawn_notes:
            continue

        # Get color based on accidental type (None for naturals)
        color = get_accidental_color(match.accidental.accidental_type)

        # Skip naturals - they don't get highlighted
        if color is None:
            continue

        drawn_notes.add(note_id)

        # Get note center
        center = (int(note.center[0]), int(note.center[1]))

        # Get note size and add 5px buffer for the ellipse
        if hasattr(note.box, 'size'):
            # Size is (width, height) of the notehead
            width = note.box.size[0]
            height = note.box.size[1]
            # Axes are half-widths with 5px buffer
            axes = (int(width / 2 + 5), int(height / 2 + 5))
        else:
            # Default size if not available
            axes = (12, 8)

        # Get rotation angle if available
        angle = 0
        if hasattr(note.box, 'angle'):
            angle = note.box.angle

        # Draw the ellipse around the notehead
        cv2.ellipse(img, center, axes, angle, 0, 360, color, 2)

    return img


def draw_accidental_note_connections_with_lines(
    image: NDArray,
    matches: list[AccidentalNoteMatch],
    staffs: list[Staff],
) -> NDArray:
    """
    Draw visualization showing which notes are affected by accidentals WITH connecting lines.

    - Draws lines from accidentals to their affected notes
    - Draws ellipses around affected notes (5px buffer around notehead)
    - Red for sharps, light blue for flats
    - Naturals are NOT highlighted

    Args:
        image: Original image to draw on
        matches: List of AccidentalNoteMatch objects
        staffs: List of Staff objects (for drawing staff lines)

    Returns:
        Image with affected notes highlighted and lines connecting them
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Track which notes we've already drawn ellipses for
    drawn_notes: set[int] = set()

    # Draw lines first (so ellipses are on top)
    for match in matches:
        # Get color based on accidental type (None for naturals)
        color = get_accidental_color(match.accidental.accidental_type)

        # Skip naturals
        if color is None:
            continue

        acc = match.accidental
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

    # Draw ellipses around each affected note
    for match in matches:
        note = match.note
        note_id = id(note)

        # Skip if we've already drawn this note
        if note_id in drawn_notes:
            continue

        # Get color based on accidental type (None for naturals)
        color = get_accidental_color(match.accidental.accidental_type)

        # Skip naturals
        if color is None:
            continue

        drawn_notes.add(note_id)

        # Get note center
        center = (int(note.center[0]), int(note.center[1]))

        # Get note size and add 5px buffer for the ellipse
        if hasattr(note.box, 'size'):
            width = note.box.size[0]
            height = note.box.size[1]
            axes = (int(width / 2 + 5), int(height / 2 + 5))
        else:
            axes = (12, 8)

        # Get rotation angle if available
        angle = 0
        if hasattr(note.box, 'angle'):
            angle = note.box.angle

        # Draw the ellipse around the notehead
        cv2.ellipse(img, center, axes, angle, 0, 360, color, 2)

    # Also draw boxes around the accidentals
    drawn_accidentals: set[int] = set()
    for match in matches:
        acc = match.accidental
        acc_id = id(acc)

        if acc_id in drawn_accidentals:
            continue

        color = get_accidental_color(acc.accidental_type)
        if color is None:
            continue

        drawn_accidentals.add(acc_id)
        acc.box.draw_onto_image(img, color)

    return img


def draw_letter_labels_visualization(
    image: NDArray,
    accidentals: list,
    notes: list,
    staffs: list[Staff],
    clef: str = "treble"
) -> NDArray:
    """
    Draw visualization showing just the letter values for accidentals and notes.

    - Accidentals show letter + accidental symbol (e.g., "F#", "Bb", "GN")
    - Notes show just the letter (e.g., "F", "G", "D")

    Args:
        image: Original image to draw on
        accidentals: List of Accidental objects
        notes: List of Note objects
        staffs: List of Staff objects
        clef: "treble" or "bass" for pitch calculation

    Returns:
        Image with letter labels
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Colors
    acc_color = (0, 0, 255)      # Red for accidentals
    note_color = (255, 0, 0)     # Blue for notes

    # Draw accidentals with letter + symbol
    for acc in accidentals:
        pitch = acc.get_pitch_name(clef)
        letter = extract_note_letter(pitch)

        # Get the accidental symbol
        symbol = acc.accidental_type.get_symbol()
        label = f"{letter}{symbol}"

        # Position label above the accidental
        x = int(acc.center[0]) - 10
        y = int(acc.center[1]) - 10

        # Draw background rectangle for readability
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), acc_color, 1)

        cv2.putText(
            img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            acc_color,
            1,
            cv2.LINE_AA
        )

    # Draw notes with just the letter
    for note in notes:
        pitch = note.get_pitch_name(clef)
        letter = extract_note_letter(pitch)

        # Position label below the note
        x = int(note.center[0]) - 5
        y = int(note.center[1]) + 20

        # Draw background rectangle for readability
        (text_w, text_h), baseline = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), (255, 255, 255), -1)
        cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), note_color, 1)

        cv2.putText(
            img,
            letter,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            note_color,
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
