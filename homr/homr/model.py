from abc import abstractmethod
from collections.abc import Callable
from enum import Enum

import cv2
import numpy as np
from typing_extensions import Self

from homr import constants
from homr.bounding_boxes import (
    AngledBoundingBox,
    BoundingBox,
    BoundingEllipse,
    DebugDrawable,
    RotatedBoundingBox,
)
from homr.type_definitions import NDArray


class InputPredictions:
    def __init__(
        self,
        original: NDArray,
        preprocessed: NDArray,
        notehead: NDArray,
        symbols: NDArray,
        staff: NDArray,
        clefs_keys: NDArray,
        stems_rest: NDArray,
    ) -> None:
        self.original = original
        self.preprocessed = preprocessed
        self.notehead = notehead
        self.symbols = symbols
        self.staff = staff
        self.stems_rest = stems_rest
        self.clefs_keys = clefs_keys


class SymbolOnStaff(DebugDrawable):
    def __init__(self, center: tuple[float, float]) -> None:
        self.center = center

    @abstractmethod
    def copy(self) -> Self:
        pass

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> Self:
        copy = self.copy()
        copy.center = transformation(self.center)
        return copy


class AccidentalType(Enum):
    SHARP = "sharp"
    FLAT = "flat"
    NATURAL = "natural"
    DOUBLE_SHARP = "double_sharp"
    DOUBLE_FLAT = "double_flat"
    KEY_SHARP = "key_sharp"
    KEY_FLAT = "key_flat"
    KEY_NATURAL = "key_natural"
    UNKNOWN = "unknown"

    @staticmethod
    def from_class_name(class_name: str) -> "AccidentalType":
        """Convert YOLOv10 class name to AccidentalType."""
        class_name_lower = class_name.lower()
        if "doublesharp" in class_name_lower:
            return AccidentalType.DOUBLE_SHARP
        elif "doubleflat" in class_name_lower:
            return AccidentalType.DOUBLE_FLAT
        elif "keysharp" in class_name_lower:
            return AccidentalType.KEY_SHARP
        elif "keyflat" in class_name_lower:
            return AccidentalType.KEY_FLAT
        elif "keynatural" in class_name_lower:
            return AccidentalType.KEY_NATURAL
        elif "sharp" in class_name_lower:
            return AccidentalType.SHARP
        elif "flat" in class_name_lower:
            return AccidentalType.FLAT
        elif "natural" in class_name_lower:
            return AccidentalType.NATURAL
        else:
            return AccidentalType.UNKNOWN

    def get_symbol(self) -> str:
        """Get the musical symbol for this accidental type."""
        symbols = {
            AccidentalType.SHARP: "#",
            AccidentalType.FLAT: "b",
            AccidentalType.NATURAL: "N",
            AccidentalType.DOUBLE_SHARP: "##",
            AccidentalType.DOUBLE_FLAT: "bb",
            AccidentalType.KEY_SHARP: "#",
            AccidentalType.KEY_FLAT: "b",
            AccidentalType.KEY_NATURAL: "N",
            AccidentalType.UNKNOWN: "?",
        }
        return symbols.get(self, "?")


class Accidental(SymbolOnStaff):
    # Class variable to track accidental indices for visualization
    _accidental_counter = 0

    def __init__(
        self,
        box: BoundingBox | RotatedBoundingBox,
        position: int,
        accidental_type: AccidentalType = AccidentalType.UNKNOWN,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(box.center)
        self.box = box
        self.position = position
        self.accidental_type = accidental_type
        self.confidence = confidence
        self.accidental_index: int | None = None  # Assigned during visualization

    @classmethod
    def reset_accidental_counter(cls) -> None:
        cls._accidental_counter = 0

    @classmethod
    def get_next_accidental_index(cls) -> int:
        cls._accidental_counter += 1
        return cls._accidental_counter

    def get_pitch_name(self, clef: str = "treble") -> str:
        """
        Convert staff position to pitch name with accidental.
        Position 0 = bottom line of staff (E4 for treble, G2 for bass)
        Each position step = one note (line or space)
        """
        # For treble clef: position 0 = E4, position increases go up
        if clef == "treble":
            base_note_index = 4  # E is index 4 in note_names (C=0, D=1, E=2, F=3, G=4, A=5, B=6)
            base_octave = 4
        else:  # bass clef
            base_note_index = 4  # G
            base_octave = 2

        # Calculate pitch from position
        note_offset = self.position + base_note_index
        octave_offset = note_offset // 7
        note_in_octave = note_offset % 7

        if note_in_octave < 0:
            note_in_octave += 7
            octave_offset -= 1

        pitch_name = note_names[note_in_octave]
        octave = base_octave + octave_offset

        # Add accidental symbol
        accidental_symbol = self.accidental_type.get_symbol()

        return f"{pitch_name}{accidental_symbol}{octave}"

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)

        # Get accidental index if not already assigned
        if self.accidental_index is None:
            self.accidental_index = Accidental.get_next_accidental_index()

        # Get pitch name from position
        pitch_name = self.get_pitch_name()

        # Format: "acc1 - F#4" or "acc2 - Bb3"
        label = f"acc{self.accidental_index} - {pitch_name}"

        # Get position for text (handle both BoundingBox and RotatedBoundingBox)
        if hasattr(self.box, 'box') and isinstance(self.box.box, (list, tuple)) and len(self.box.box) >= 2:
            if isinstance(self.box.box[0], (int, float)):
                text_x = int(self.box.box[0])
                text_y = int(self.box.box[1])
            else:
                # RotatedBoundingBox format: ((center_x, center_y), (width, height), angle)
                text_x = int(self.box.center[0])
                text_y = int(self.box.center[1])
        else:
            text_x = int(self.center[0])
            text_y = int(self.center[1])

        cv2.putText(
            img,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return f"Accidental({self.center}, {self.accidental_type.value}, pos={self.position})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Accidental":
        return Accidental(self.box, self.position, self.accidental_type, self.confidence)


class Rest(SymbolOnStaff):
    def __init__(self, box: BoundingBox) -> None:
        super().__init__(box.center)
        self.box = box
        self.has_dot = False

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            "rest",
            (int(self.box.box[0]), int(self.box.box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "Rest(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Rest":
        return Rest(self.box)


class StemDirection(Enum):
    UP = 1
    DOWN = 2


class NoteHeadType(Enum):
    HOLLOW = 1
    SOLID = 2

    def __str__(self) -> str:
        if self == NoteHeadType.HOLLOW:
            return "O"
        elif self == NoteHeadType.SOLID:
            return "*"
        else:
            raise Exception("Unknown NoteHeadType")


note_names = ["C", "D", "E", "F", "G", "A", "B"]


class Note(SymbolOnStaff):
    # Class variable to track note indices for visualization
    _note_counter = 0

    def __init__(
        self,
        box: BoundingEllipse,
        position: int,
        stem: RotatedBoundingBox | None,
        stem_direction: StemDirection | None,
    ):
        super().__init__(box.center)
        self.box = box
        self.position = position
        self.has_dot = False
        self.stem = stem
        self.circle_of_fifth = 0
        self.stem_direction = stem_direction
        self.beams: list[RotatedBoundingBox] = []
        self.flags: list[RotatedBoundingBox] = []
        self.note_index: int | None = None  # Assigned during visualization

    @classmethod
    def reset_note_counter(cls) -> None:
        cls._note_counter = 0

    @classmethod
    def get_next_note_index(cls) -> int:
        cls._note_counter += 1
        return cls._note_counter

    def get_pitch_name(self, clef: str = "treble") -> str:
        """
        Convert staff position to pitch name.
        Position 0 = bottom line of staff (E4 for treble, G2 for bass)
        Each position step = one note (line or space)
        """
        # For treble clef: position 0 = E4, position increases go up
        # Staff positions: 0=E4, 1=F4, 2=G4, 3=A4, 4=B4, 5=C5, 6=D5, 7=E5, 8=F5
        if clef == "treble":
            base_note_index = 4  # E is index 4 in note_names (C=0, D=1, E=2, F=3, G=4, A=5, B=6)
            base_octave = 4
        else:  # bass clef
            base_note_index = 4  # G is index 4... wait, G=4 in (C,D,E,F,G,A,B)
            base_octave = 2

        # Calculate pitch from position
        # position 0 = bottom staff line
        note_offset = self.position + base_note_index
        octave_offset = note_offset // 7
        note_in_octave = note_offset % 7

        if note_in_octave < 0:
            note_in_octave += 7
            octave_offset -= 1

        pitch_name = note_names[note_in_octave]
        octave = base_octave + octave_offset

        return f"{pitch_name}{octave}"

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        # Only draw the notehead ellipse, not stems/beams/flags
        self.box.draw_onto_image(img, color)

        # Get note index if not already assigned
        if self.note_index is None:
            self.note_index = Note.get_next_note_index()

        # Get pitch name from position
        pitch_name = self.get_pitch_name()

        # Format: "note1 - C4"
        label = f"note{self.note_index} - {pitch_name}"

        cv2.putText(
            img,
            label,
            (int(self.box.center[0]), int(self.box.center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "Note(" + str(self.center) + ", " + str(self.position) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Note":
        return Note(self.box, self.position, self.stem, self.stem_direction)


class BarLine(SymbolOnStaff):
    def __init__(self, box: RotatedBoundingBox):
        super().__init__(box.center)
        self.box = box

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)

    def __str__(self) -> str:
        return "BarLine(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "BarLine":
        return BarLine(self.box)


class Clef(SymbolOnStaff):
    def __init__(self, box: BoundingBox):
        super().__init__(box.center)
        self.box = box

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        self.box.draw_onto_image(img, color)
        cv2.putText(
            img,
            "clef",
            (self.box.box[0], self.box.box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

    def __str__(self) -> str:
        return "Clef(" + str(self.center) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Clef":
        return Clef(self.box)


class StaffPoint:
    def __init__(self, x: float, y: list[float], angle: float):
        if len(y) % constants.number_of_lines_on_a_staff != 0:
            raise Exception("A staff must consist of 5, 10, ... lines")
        self.x = x
        self.y = y
        self.angle = angle
        self.average_unit_size = np.mean(np.diff(y))

    def merge(self, other: "StaffPoint") -> "StaffPoint":
        if abs(self.x - other.x) > 1e-3:
            raise ValueError("Can't merge points at different positions")
        y = []
        y.extend(self.y)
        y.extend(other.y)
        angle = (self.angle + other.angle) / 2
        return StaffPoint(self.x, sorted(y), angle)

    def find_position_in_unit_sizes(self, box: AngledBoundingBox) -> int:
        center = box.center
        idx_of_closest_y = int(np.argmin(np.abs([y_value - center[1] for y_value in self.y])))
        distance = self.y[idx_of_closest_y] - center[1]
        distance_in_unit_sizes = round(2 * distance / self.average_unit_size)
        position = 2 * (len(self.y) - idx_of_closest_y) + distance_in_unit_sizes - 1
        return position

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> "StaffPoint":
        xy = [transformation((self.x, y_value)) for y_value in self.y]
        average_x = np.mean([x for x, _ in xy])
        return StaffPoint(float(average_x), [y for _, y in xy], self.angle)

    def to_bounding_box(self) -> BoundingBox:
        return BoundingBox(
            [int(self.x), int(self.y[0]), int(self.x), int(self.y[-1])], np.array([]), -2
        )

    def __str__(self) -> str:
        return "P(" + str(self.x) + "," + str(self.y[2]) + ")"

    def __repr__(self) -> str:
        return str(self)


class Staff(DebugDrawable):
    def __init__(self, grid: list[StaffPoint]):
        self.grid = grid
        self.min_x = grid[0].x
        self.max_x = grid[-1].x
        self.min_y = min([min(p.y) for p in grid])
        self.max_y = max([max(p.y) for p in grid])
        self.average_unit_size = np.median([p.average_unit_size for p in grid])
        self.symbols: list[SymbolOnStaff] = []
        self.is_grandstaff = False
        self._y_tolerance = constants.max_number_of_ledger_lines * self.average_unit_size

    def is_on_staff_zone(self, item: AngledBoundingBox) -> bool:
        point = self.get_at(item.center[0])
        if point is None:
            return False
        if (
            item.center[1] > point.y[-1] + self._y_tolerance
            or item.center[1] < point.y[0] - self._y_tolerance
        ):
            return False
        return True

    def merge(self, other: "Staff") -> "Staff":
        grid_a: dict[int, StaffPoint] = {}
        for p in self.grid:
            grid_a[int(round(p.x))] = p
        grid_b: dict[int, StaffPoint] = {}
        for p in other.grid:
            grid_b[int(round(p.x))] = p
        x_positions = set(grid_a.keys()).intersection(grid_b.keys())

        grid = [grid_a[x].merge(grid_b[x]) for x in sorted(x_positions)]
        result = Staff(grid)
        result.symbols.extend(self.symbols)
        result.symbols.extend(other.symbols)
        result.is_grandstaff = True
        return result

    def add_symbol(self, symbol: SymbolOnStaff) -> None:
        self.symbols.append(symbol)

    def get_at(self, x: float) -> StaffPoint | None:
        closest_point = min(self.grid, key=lambda p: abs(p.x - x))
        if abs(closest_point.x - x) > constants.staff_position_tolerance:
            return None
        return closest_point

    def y_distance_to(self, point: tuple[float, float]) -> float:
        staff_point = self.get_at(point[0])
        if staff_point is None:
            return 1e10  # Something large to mimic infinity
        return min([abs(y - point[1]) for y in staff_point.y])

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        if len(self.grid) == 0:
            return
        for i in range(len(self.grid[0].y)):
            for j in range(len(self.grid) - 1):
                p1 = self.grid[j]
                p2 = self.grid[j + 1]
                cv2.line(
                    img, (int(p1.x), int(p1.y[i])), (int(p2.x), int(p2.y[i])), color, thickness=2
                )

    def get_bar_lines(self) -> list[BarLine]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, BarLine):
                result.append(symbol)
        return result

    def get_clefs(self) -> list[Clef]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, Clef):
                result.append(symbol)
        return result

    def get_notes(self) -> list[Note]:
        result = []
        for symbol in self.symbols:
            if isinstance(symbol, Note):
                result.append(symbol)
        return result

    def extend_to_x_range(self, min_x: int, max_x: int) -> "Staff":
        grid = self.grid.copy()

        if min_x >= 0 and min_x < grid[0].x:
            grid.insert(0, StaffPoint(min_x, grid[0].y, grid[0].angle))
        if max_x >= 0 and max_x > grid[-1].x:
            grid.append(StaffPoint(max_x, grid[-1].y, grid[-1].angle))

        return Staff(grid)

    def get_number_of_notes(self) -> int:
        result = 0
        for symbol in self.symbols:
            if isinstance(symbol, Note):
                result += 1
        return result

    def get_all_except_notes(self) -> list[SymbolOnStaff]:
        result = []
        for symbol in self.symbols:
            if not isinstance(symbol, Note):
                result.append(symbol)
        return result

    def __str__(self) -> str:
        return "Staff(" + str.join(", ", [str(s) for s in self.symbols]) + ")"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> "Staff":
        return Staff(self.grid)

    def transform_coordinates(
        self, transformation: Callable[[tuple[float, float]], tuple[float, float]]
    ) -> "Staff":
        copy = Staff([point.transform_coordinates(transformation) for point in self.grid])
        copy.symbols = [symbol.transform_coordinates(transformation) for symbol in self.symbols]
        return copy


class MultiStaff(DebugDrawable):
    """
    A grand staff or a staff with multiple voices.
    """

    def __init__(self, staffs: list[Staff], connections: list[RotatedBoundingBox]) -> None:
        self.staffs = sorted(staffs, key=lambda s: s.min_y)
        self.connections = connections

    def merge(self, other: "MultiStaff") -> "MultiStaff":
        unique_staffs = []
        unique_connections = []
        for staff in self.staffs + other.staffs:
            if staff not in unique_staffs:
                unique_staffs.append(staff)
        for connection in self.connections + other.connections:
            if connection not in unique_connections:
                unique_connections.append(connection)
        return MultiStaff(unique_staffs, unique_connections)

    def create_grandstaffs(self) -> "MultiStaff":
        if len(self.staffs) == 0:
            return self
        merged = self.staffs[0]
        for staff in self.staffs[1:]:
            merged = merged.merge(staff)
        return MultiStaff([merged], self.connections)

    def break_apart(self) -> list["MultiStaff"]:
        return [MultiStaff([staff], []) for staff in self.staffs]

    def draw_onto_image(self, img: NDArray, color: tuple[int, int, int] = (255, 0, 0)) -> None:
        for staff in self.staffs:
            staff.draw_onto_image(img, color)
        for connection in self.connections:
            connection.draw_onto_image(img, color)
