import glob
import os
from collections.abc import Sequence
from itertools import chain

import cv2

from homr.bounding_boxes import DebugDrawable
from homr.type_definitions import NDArray


class Debug:
    def __init__(self, original_image: NDArray, filename: str, debug: bool):
        self.filename = filename
        self.original_image = original_image
        filename = filename.replace("\\", "/")
        self.dir_name = os.path.dirname(filename)
        self.base_filename = os.path.join(self.dir_name, filename.split("/")[-1].split(".")[0])
        self.debug = debug
        self.colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 165, 0),
            (255, 182, 193),
            (128, 0, 128),
            (64, 224, 208),
        ]
        self.debug_output_counter = 0
        self.written_files: list[str] = []

    def clean_debug_files_from_previous_runs(self) -> None:
        prefixes = (
            self.base_filename + "_debug_",
            self.base_filename + "_tesseract_input",
            self.base_filename + "_staff-",
        )

        for file in glob.glob(self.base_filename + "*"):
            if file.startswith(prefixes) and file not in self.written_files:
                os.remove(file)

    def _debug_file_name(self, suffix: str) -> str:
        self.debug_output_counter += 1
        return f"{self.base_filename}_debug_{str(self.debug_output_counter)}_{suffix}.png"

    def write_threshold_image(self, suffix: str, image: NDArray) -> None:
        if not self.debug:
            return
        filename = self._debug_file_name(suffix)
        self._remember_file_name(filename)
        cv2.imwrite(filename, 255 * image)

    def _remember_file_name(self, filename: str) -> None:
        self.written_files.append(filename)

    def write_bounding_boxes(self, suffix: str, bounding_boxes: Sequence[DebugDrawable]) -> None:
        if not self.debug:
            return
        img = self.original_image.copy()
        for box in bounding_boxes:
            box.draw_onto_image(img)
        filename = self._debug_file_name(suffix)
        self._remember_file_name(filename)
        cv2.imwrite(filename, img)

    def write_image(self, suffix: str, image: NDArray) -> None:
        if not self.debug:
            return
        filename = self._debug_file_name(suffix)
        self._remember_file_name(filename)
        cv2.imwrite(filename, image)

    def write_image_with_fixed_suffix(self, suffix: str, image: NDArray) -> None:
        if not self.debug:
            return
        filename = self.base_filename + suffix
        self._remember_file_name(filename)
        cv2.imwrite(filename, image)

    def write_all_bounding_boxes_alternating_colors(
        self, suffix: str, *boxes: Sequence[DebugDrawable]
    ) -> None:
        self.write_bounding_boxes_alternating_colors(suffix, list(chain.from_iterable(boxes)))

    def write_bounding_boxes_alternating_colors(
        self, suffix: str, bounding_boxes: Sequence[DebugDrawable]
    ) -> None:
        if not self.debug:
            return
        self.write_teaser(self._debug_file_name(suffix), bounding_boxes)

    def write_teaser(self, filename: str, bounding_boxes: Sequence[DebugDrawable]) -> None:
        img = self.original_image.copy()
        for i, box in enumerate(bounding_boxes):
            color = self.colors[i % len(self.colors)]
            box.draw_onto_image(img, color)
        self._remember_file_name(filename)
        cv2.imwrite(filename, img)

    def write_notes_visualization(
        self, multi_staffs: Sequence[DebugDrawable], notes: Sequence[DebugDrawable]
    ) -> None:
        """Write visualization showing detected notes on original image."""
        # Late import to avoid circular dependency
        from homr.model import Note

        img = self.original_image.copy()

        # Reset note counter for consistent numbering
        Note.reset_note_counter()

        # Draw staff lines first (in gray for reference)
        for staff in multi_staffs:
            staff.draw_onto_image(img, (128, 128, 128))

        # Sort notes by x position (left to right) for sequential numbering
        sorted_notes = sorted(notes, key=lambda n: n.center[0] if hasattr(n, 'center') else 0)

        # Draw notes with alternating colors
        for i, note in enumerate(sorted_notes):
            color = self.colors[i % len(self.colors)]
            note.draw_onto_image(img, color)

        filename = self.base_filename + "_notes.png"
        cv2.imwrite(filename, img)

    def write_accidentals_visualization(
        self, multi_staffs: Sequence[DebugDrawable], accidentals: Sequence[DebugDrawable]
    ) -> None:
        """Write visualization showing detected accidentals with pitch names."""
        # Late import to avoid circular dependency
        from homr.model import Accidental, AccidentalType

        img = self.original_image.copy()

        # Reset accidental counter for consistent numbering
        Accidental.reset_accidental_counter()

        # Draw staff lines first (in gray for reference)
        for staff in multi_staffs:
            staff.draw_onto_image(img, (128, 128, 128))

        # Sort accidentals by x position (left to right) for sequential numbering
        sorted_accidentals = sorted(
            accidentals, key=lambda a: a.center[0] if hasattr(a, 'center') else 0
        )

        # Color map for different accidental types
        accidental_colors = {
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

        # Draw accidentals with type-based colors
        for acc in sorted_accidentals:
            if hasattr(acc, 'accidental_type'):
                color = accidental_colors.get(acc.accidental_type, (128, 128, 128))
            else:
                color = (128, 128, 128)
            acc.draw_onto_image(img, color)

        filename = self.base_filename + "_accidentals.png"
        cv2.imwrite(filename, img)
        print(f"Saved accidentals visualization: {filename}")

    def write_full_visualization(
        self,
        multi_staffs: Sequence[DebugDrawable],
        notes: Sequence[DebugDrawable],
        accidentals: Sequence[DebugDrawable],
    ) -> None:
        """Write visualization showing both notes AND accidentals with pitch names."""
        # Late import to avoid circular dependency
        from homr.model import Note, Accidental, AccidentalType

        img = self.original_image.copy()

        # Reset counters for consistent numbering
        Note.reset_note_counter()
        Accidental.reset_accidental_counter()

        # Draw staff lines first (in gray for reference)
        for staff in multi_staffs:
            staff.draw_onto_image(img, (128, 128, 128))

        # Sort notes by x position
        sorted_notes = sorted(notes, key=lambda n: n.center[0] if hasattr(n, 'center') else 0)

        # Sort accidentals by x position
        sorted_accidentals = sorted(
            accidentals, key=lambda a: a.center[0] if hasattr(a, 'center') else 0
        )

        # Draw notes with alternating colors
        for i, note in enumerate(sorted_notes):
            color = self.colors[i % len(self.colors)]
            note.draw_onto_image(img, color)

        # Color map for accidentals
        accidental_colors = {
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

        # Draw accidentals
        for acc in sorted_accidentals:
            if hasattr(acc, 'accidental_type'):
                color = accidental_colors.get(acc.accidental_type, (128, 128, 128))
            else:
                color = (128, 128, 128)
            acc.draw_onto_image(img, color)

        filename = self.base_filename + "_full.png"
        cv2.imwrite(filename, img)
        print(f"Saved full visualization (notes + accidentals): {filename}")

    def write_accidental_effects_visualization(
        self,
        multi_staffs: Sequence[DebugDrawable],
        notes: Sequence[DebugDrawable],
        accidentals: Sequence[DebugDrawable],
        staffs: list,
    ) -> None:
        """
        Write visualization showing which notes are affected by which accidentals.
        Draws lines connecting accidentals to their affected notes and circles affected notes.
        """
        from homr.accidental_note_matching import (
            match_accidentals_to_notes,
            draw_accidental_note_connections,
            create_accidental_effects_summary,
        )
        from homr.model import Accidental, Note

        # Convert to proper types
        acc_list = [a for a in accidentals if isinstance(a, Accidental)]
        note_list = [n for n in notes if isinstance(n, Note)]

        # Match accidentals to notes
        matches = match_accidentals_to_notes(staffs, acc_list, note_list)

        # Draw visualization
        img = draw_accidental_note_connections(self.original_image, matches, staffs)

        filename = self.base_filename + "_accidental_effects.png"
        cv2.imwrite(filename, img)
        print(f"Saved accidental effects visualization: {filename}")

        # Print summary
        summary = create_accidental_effects_summary(matches)
        print(summary)

    def write_model_input_image(self, suffix: str, staff_image: NDArray) -> str:
        """
        These files aren't really debug files, but it's convenient to handle them here
        so that they are cleaned up together with the debug files.

        Model input images are the input to the transformer or OCR images.
        """
        filename = self.base_filename + suffix
        if self.debug:
            self._remember_file_name(filename)
        cv2.imwrite(filename, staff_image)
        return filename
