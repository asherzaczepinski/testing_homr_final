import argparse
import glob
import os
import sys
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort

from homr import color_adjust, download_utils
from homr.accidental_detection import detect_and_position_accidentals
from homr.autocrop import autocrop
from homr.bar_line_detection import (
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug
from homr.model import Accidental, InputPredictions, MultiStaff, Note, Staff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.resize import resize_image
from homr.segmentation.config import segnet_path_onnx, segnet_path_onnx_fp16
from homr.segmentation.inference_segnet import extract
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.staff_position_save_load import load_staff_positions, save_staff_positions
from homr.title_detection import detect_title, download_ocr_weights
from homr.transformer.configs import Config, default_config
from homr.type_definitions import NDArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class AccidentalDetection:
    """Represents a detected accidental with its class name."""
    bbox: RotatedBoundingBox
    class_name: str
    confidence: float


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
        accidentals: list[AccidentalDetection] | None = None,
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines
        self.accidentals = accidentals or []


def calculate_overlap_percentage(box1, box2):
    """Calculate what percentage of box1 overlaps with box2."""
    # Get bounding rectangles
    x1_min, y1_min = box1.top_left
    x1_max, y1_max = box1.bottom_right
    x2_min, y2_min = box2.top_left
    x2_max, y2_max = box2.bottom_right

    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = x_overlap * y_overlap

    # Calculate box1's area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)

    if box1_area == 0:
        return 0

    return overlap_area / box1_area


class InvalidProgramArgumentException(Exception):
    """Raise this exception for issues which the user can address."""


class GpuSupport(Enum):
    No = "no"
    AUTO = "auto"
    FORCE = "force"


def get_predictions(
    original: NDArray,
    preprocessed: NDArray,
    img_path: str,
    enable_cache: bool,
    use_gpu_inference: bool,
) -> InputPredictions:
    result = extract(
        preprocessed,
        img_path,
        step_size=320,
        use_cache=enable_cache,
        use_gpu_inference=use_gpu_inference,
    )
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool, use_gpu_inference: bool, debug_base_path: str = ""
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidProgramArgumentException(
            "The file format is not supported, please provide a JPG or PNG image file:" + image_path
        )
    image = autocrop(image)
    image = resize_image(image)
    preprocessed, _background = color_adjust.color_adjust(image, 40)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache, use_gpu_inference)
    # Use debug_base_path for output files if provided, otherwise use image_path
    debug_path = debug_base_path if debug_base_path else image_path
    debug = Debug(predictions.original, debug_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def predict_symbols(debug: Debug, predictions: InputPredictions, unit_size: float = 0) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    # Detect all symbols from clefs_keys prediction
    eprint("Creating bounds for clefs_keys and accidentals")
    all_symbols = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(10, 15), max_size=(1000, 1000), skip_merging=True
    )

    # Separate accidentals from clefs based on size
    # Size filtering is RELATIVE to staff line spacing (unit_size):
    # - Accidentals: max 2 line spacings wide, max 3 line spacings tall
    # - Clefs: typically 4+ line spacings tall
    accidentals_only = []
    clefs_only = []

    # Calculate size limits based on unit_size
    if unit_size > 0:
        # Relative to staff line spacing
        min_acc_width = int(unit_size * 0.3)
        max_acc_width = int(unit_size * 2.0)    # Max 2 line spacings
        min_acc_height = int(unit_size * 0.8)
        max_acc_height = int(unit_size * 3.0)   # Max 3 line spacings
        min_clef_height = int(unit_size * 3.5)  # Clefs are taller than accidentals
        eprint(f"Geometric filtering (unit_size={unit_size:.1f}px): accidentals w:{min_acc_width}-{max_acc_width}, h:{min_acc_height}-{max_acc_height}")
    else:
        # Fallback to fixed pixel values
        min_acc_width, max_acc_width = 5, 25
        min_acc_height, max_acc_height = 15, 35
        min_clef_height = 38
        eprint("Geometric filtering (fallback): accidentals w:5-25, h:15-35")

    for symbol in all_symbols:
        height = symbol.size[1]
        width = symbol.size[0]
        # Accidentals: relative to staff line spacing
        if min_acc_height <= height <= max_acc_height and min_acc_width <= width <= max_acc_width:
            accidentals_only.append(symbol)
        elif height >= min_clef_height:
            clefs_only.append(symbol)
        # Anything else (too small, too big, wrong proportions) is ignored

    eprint(f"Found {len(all_symbols)} symbols: {len(accidentals_only)} potential accidentals, {len(clefs_only)} clefs")

    # Filter out accidentals that overlap >50% with clefs
    filtered_accidentals = []
    for accidental in accidentals_only:
        overlaps_clef = False
        for clef in clefs_only:
            overlap_pct = calculate_overlap_percentage(accidental, clef)
            if overlap_pct > 0.5:
                overlaps_clef = True
                break

        if not overlaps_clef:
            filtered_accidentals.append(accidental)

    removed_overlap = len(accidentals_only) - len(filtered_accidentals)
    if removed_overlap > 0:
        eprint(f"Removed {removed_overlap} accidentals overlapping with clefs (>50%)")
    eprint(f"Final: {len(filtered_accidentals)} accidentals after overlap filtering")

    # Also filter overlapping accidentals among themselves (keep higher confidence or first one)
    final_accidentals = []
    for i, acc in enumerate(filtered_accidentals):
        should_keep = True
        for j, other_acc in enumerate(filtered_accidentals):
            if i >= j:  # Only compare with earlier ones (already kept)
                continue
            overlap_pct = calculate_overlap_percentage(acc, other_acc)
            if overlap_pct > 0.5:
                should_keep = False
                break
        if should_keep:
            final_accidentals.append(acc)

    removed_self_overlap = len(filtered_accidentals) - len(final_accidentals)
    if removed_self_overlap > 0:
        eprint(f"Removed {removed_self_overlap} overlapping accidentals (>50% overlap with each other)")

    # Use clefs_only for clefs_keys (for staff detection)
    # Use final_accidentals for accidentals (stored separately)
    clefs_keys = clefs_only

    # Create AccidentalDetection objects (with geometric fallback classification)
    accidental_detections = []
    for acc_box in final_accidentals:
        # Classify by aspect ratio (geometric fallback)
        height = acc_box.size[1]
        width = acc_box.size[0]
        if width > 0:
            aspect_ratio = height / width
            if aspect_ratio > 2.5:
                class_name = "accidentalSharp"
            elif aspect_ratio > 1.8:
                class_name = "accidentalFlat"
            else:
                class_name = "accidentalNatural"
        else:
            class_name = "unknown"

        accidental_detections.append(AccidentalDetection(
            bbox=acc_box,
            class_name=class_name,
            confidence=1.0  # Geometric detection has no confidence score
        ))

    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines, accidental_detections)


def get_accidental_model_path() -> str:
    """Get the path to the accidental detection model."""
    return os.path.join(os.path.dirname(__file__), "models", "accidentals", "best.pt")


@dataclass
class ProcessingConfig:
    enable_debug: bool
    enable_cache: bool
    write_staff_positions: bool
    read_staff_positions: bool
    selected_staff: int
    use_gpu_inference: bool
    visualize: bool
    detect_accidentals: bool = True


def process_image(
    image_path: str,
    config: ProcessingConfig,
    xml_generator_args: XmlGeneratorArguments,
    output_folder: str = "",
) -> None:
    eprint("Processing " + image_path)
    xml_file = get_output_path(image_path, output_folder, ".musicxml")
    debug_cleanup: Debug | None = None
    notes: list[Note] = []
    accidentals: list[Accidental] = []
    staffs: list[Staff] = []
    original_image: NDArray | None = None
    try:
        if config.read_staff_positions:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read " + image_path)
            image = resize_image(image)
            # For debug, use output folder path if provided
            debug_base_path = get_output_path(image_path, output_folder, "") if output_folder else image_path
            debug = Debug(image, debug_base_path, config.enable_debug)
            staff_position_files = replace_extension(image_path, ".txt")
            multi_staffs = load_staff_positions(
                debug, image, staff_position_files, config.selected_staff
            )
            title = ""
        else:
            multi_staffs, image, debug, title_future, notes, staffs, original_image = detect_staffs_in_image(image_path, config, output_folder)
        debug_cleanup = debug

        transformer_config = Config()
        transformer_config.use_gpu_inference = config.use_gpu_inference

        result_staffs = parse_staffs(
            debug,
            multi_staffs,
            image,
            selected_staff=config.selected_staff,
            config=transformer_config,
        )

        title = title_future.result(60)
        eprint("Found title:", title)

        eprint("Writing XML", result_staffs)
        xml = generate_xml(xml_generator_args, result_staffs, title)
        xml.write(xml_file)

        eprint("Finished parsing " + str(len(result_staffs)) + " staves")
        teaser_file = get_output_path(image_path, output_folder, "_teaser.png")
        if config.write_staff_positions:
            staff_position_files = get_output_path(image_path, output_folder, ".txt")
            save_staff_positions(multi_staffs, image.shape, staff_position_files)
        debug.write_teaser(teaser_file, multi_staffs)

        # Detect accidentals with YOLOv10
        if config.detect_accidentals and len(staffs) > 0 and original_image is not None:
            model_path = get_accidental_model_path()
            if os.path.exists(model_path):
                eprint("Detecting accidentals with YOLOv10...")
                accidentals = detect_and_position_accidentals(
                    image=original_image,
                    staffs=staffs,
                    model_path=model_path,
                    confidence_threshold=0.3,
                )
                eprint(f"Found {len(accidentals)} accidentals")
            else:
                eprint("Accidental detection model not found, skipping accidental detection")

        # Write visualizations
        if config.visualize:
            if len(notes) > 0:
                debug.write_notes_visualization(multi_staffs, notes)
                eprint("Notes visualization written to", get_output_path(image_path, output_folder, "_notes.png"))

            if len(accidentals) > 0:
                debug.write_accidentals_visualization(multi_staffs, accidentals)
                eprint("Accidentals visualization written to", get_output_path(image_path, output_folder, "_accidentals.png"))

                # Write combined visualization
                debug.write_full_visualization(multi_staffs, notes, accidentals)
                eprint("Full visualization written to", get_output_path(image_path, output_folder, "_full.png"))

                # Write accidental effects visualization (which notes are affected by which accidentals)
                if len(notes) > 0 and len(staffs) > 0:
                    debug.write_accidental_effects_visualization(multi_staffs, notes, accidentals, staffs)
                    eprint("Accidental effects visualization written to", get_output_path(image_path, output_folder, "_accidental_effects.png"))

        debug.clean_debug_files_from_previous_runs()

        eprint("Result was written to", xml_file)
    except:
        if os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        if debug_cleanup is not None:
            debug_cleanup.clean_debug_files_from_previous_runs()


def detect_staffs_in_image(
    image_path: str, config: ProcessingConfig, output_folder: str = ""
) -> tuple[list[MultiStaff], NDArray, Debug, Future[str], list[Note], list[Staff], NDArray]:
    # Use output folder for debug files if provided
    debug_base_path = get_output_path(image_path, output_folder, "") if output_folder else image_path
    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache, config.use_gpu_inference, debug_base_path
    )
    symbols = predict_symbols(debug, predictions)

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
    eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
    eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")
    if len(noteheads_with_stems) == 0:
        raise Exception("No noteheads found")

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    eprint("Average note head height: " + str(average_note_head_height))

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
    eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

    debug.write_bounding_boxes(
        "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
    )
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    if len(staffs) == 0:
        raise Exception("No staffs found")
    title_future = detect_title(debug, staffs[0])
    debug.write_bounding_boxes_alternating_colors("staffs", staffs)

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    debug.write_threshold_image("brace_dot", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    debug.write_all_bounding_boxes_alternating_colors("notes", multi_staffs, notes)

    return multi_staffs, predictions.preprocessed, debug, title_future, notes, staffs, predictions.original


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


def download_weights(use_gpu_inference: bool) -> None:
    base_url = "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/"
    if use_gpu_inference:
        models = [
            segnet_path_onnx_fp16,
            default_config.filepaths.encoder_path_fp16,
            default_config.filepaths.decoder_path_fp16,
        ]
        missing_models = [model for model in models if not os.path.exists(model)]
    else:
        models = [
            segnet_path_onnx,
            default_config.filepaths.encoder_path,
            default_config.filepaths.decoder_path,
        ]
        missing_models = [model for model in models if not os.path.exists(model)]

    if len(missing_models) == 0:
        return

    eprint("Downloading", len(missing_models), "models - this is only required once")
    for model in missing_models:
        if not os.path.exists(model):
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            try:
                zip_name = base_name + ".zip"
                download_url = base_url + zip_name
                downloaded_zip = os.path.join(os.path.dirname(model), zip_name)
                download_utils.download_file(download_url, downloaded_zip)

                destination_dir = os.path.dirname(model)
                download_utils.unzip_file(downloaded_zip, destination_dir)
            finally:
                if os.path.exists(downloaded_zip):
                    os.remove(downloaded_zip)


def setup_output_folder(input_path: str) -> str:
    """
    Create/clear an output folder for processed files.

    For folder input: creates 'output' folder next to input folder
    For single file: returns None (outputs go next to input file)

    Returns:
        Path to output folder, or empty string for single file mode
    """
    if os.path.isdir(input_path):
        # Create output folder next to the input folder
        parent_dir = os.path.dirname(os.path.abspath(input_path))
        output_folder = os.path.join(parent_dir, "output")

        # Clear the output folder if it exists
        if os.path.exists(output_folder):
            import shutil
            shutil.rmtree(output_folder)
            eprint(f"Cleared existing output folder: {output_folder}")

        # Create fresh output folder
        os.makedirs(output_folder, exist_ok=True)
        eprint(f"Output folder created: {output_folder}")
        return output_folder
    else:
        # Single file mode - outputs go next to the input file
        return ""


def get_output_path(input_path: str, output_folder: str, new_extension: str) -> str:
    """
    Get the output path for a file.

    Args:
        input_path: Original input file path
        output_folder: Output folder path (empty string for single file mode)
        new_extension: New file extension (e.g., ".musicxml", "_notes.png")

    Returns:
        Full path to output file
    """
    base_name = os.path.basename(input_path)
    name_without_ext = os.path.splitext(base_name)[0]

    if output_folder:
        # Folder mode - output to output folder
        # Include a fake extension for Debug class to strip
        if new_extension == "":
            return os.path.join(output_folder, name_without_ext + ".png")
        return os.path.join(output_folder, name_without_ext + new_extension)
    else:
        # Single file mode - output next to input file
        return replace_extension(input_path, new_extension)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="homer", description="An optical music recognition (OMR) system"
    )
    parser.add_argument("image", type=str, nargs="?", help="Path to the image to process")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Downloads the models if they are missing and then exits. "
        + "You don't have to call init before processing images, "
        + "it's only useful if you want to prepare for example a Docker image.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--cache", action="store_true", help="Read an existing cache file or create a new one"
    )
    parser.add_argument(
        "--output-large-page",
        action="store_true",
        help="Adds instructions to the musicxml so that it gets rendered on larger pages",
    )
    parser.add_argument(
        "--output-metronome", type=int, help="Adds a metronome to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--output-tempo", type=int, help="Adds a tempo to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--write-staff-positions",
        action="store_true",
        help="Writes the position of all detected staffs to a txt file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Output a visualization image showing detected notes overlaid on the original",
    )
    parser.add_argument(
        "--no-accidentals",
        action="store_true",
        help="Disable YOLOv10 accidental detection",
    )
    parser.add_argument(
        "--read-staff-positions",
        action="store_true",
        help="Reads the position of all staffs from a txt file instead"
        + " of running the built-in staff detection.",
    )
    parser.add_argument(
        "--gpu",
        type=GpuSupport,
        choices=list(GpuSupport),
        default=GpuSupport.AUTO,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    has_gpu_support = "CUDAExecutionProvider" in ort.get_available_providers()

    use_gpu_inference = (
        args.gpu == GpuSupport.AUTO and has_gpu_support
    ) or args.gpu == GpuSupport.FORCE

    download_weights(use_gpu_inference)
    if args.init:
        download_ocr_weights()
        eprint("Init finished")
        return

    config = ProcessingConfig(
        args.debug,
        args.cache,
        args.write_staff_positions,
        args.read_staff_positions,
        -1,
        use_gpu_inference,
        args.visualize,
        detect_accidentals=not args.no_accidentals,
    )

    xml_generator_args = XmlGeneratorArguments(
        args.output_large_page, args.output_metronome, args.output_tempo
    )

    if not args.image:
        eprint("No image provided")
        parser.print_help()
        sys.exit(1)
    elif os.path.isfile(args.image):
        try:
            process_image(args.image, config, xml_generator_args, output_folder="")
        except InvalidProgramArgumentException as e:
            eprint(str(e))
            sys.exit(2)
    elif os.path.isdir(args.image):
        # Setup output folder (clears existing one)
        output_folder = setup_output_folder(args.image)

        image_files = get_all_image_files_in_folder(args.image)
        eprint("Processing", len(image_files), "files:", image_files)
        error_files = []
        for image_file in image_files:
            eprint("=========================================")
            try:
                process_image(image_file, config, xml_generator_args, output_folder=output_folder)
                eprint("Finished", image_file)
            except Exception as e:
                eprint(f"An error occurred while processing {image_file}: {e}")
                error_files.append(image_file)
        if len(error_files) > 0:
            eprint("Errors occurred while processing the following files:", error_files)
    else:
        eprint(f"{args.image} is not a valid file or directory")
        sys.exit(2)


if __name__ == "__main__":
    main()
