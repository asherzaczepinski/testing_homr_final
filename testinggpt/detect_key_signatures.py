#!/usr/bin/env python3
"""
PRODUCTION VERSION: GPT-4o Key Signature Detection
Integrates with Orchestra-AI-2 pipeline to:
1. Run Orchestra-AI-2 to extract first measures with staff lines removed
2. Detect key signatures using GPT-4o
3. Save results
"""

import os
import sys
import json
import base64
import subprocess
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def prepare_image(image_path, max_size_mb=18):
    """Prepare high-quality PNG under 20MB GPT limit"""
    path = Path(image_path)
    img = Image.open(image_path)
    temp_path = path.parent / f"temp_{path.stem}.png"

    for compress_level in range(1, 6):
        img.save(temp_path, 'PNG', compress_level=compress_level)
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        if size_mb <= max_size_mb:
            return str(temp_path), size_mb

    return str(temp_path), size_mb


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def detect_key_signatures(image_path, model="gpt-4o", verbose=True):
    """
    Detect key signatures from sheet music image

    Args:
        image_path: Path to sheet music image
        model: GPT model to use (default: gpt-4o)
        verbose: Print progress messages

    Returns:
        list: Key signatures for each staff, e.g., ["G", "G", "G"]
        Returns None on error
    """
    if verbose:
        print(f"Analyzing: {Path(image_path).name}")

    # Prepare image
    hq_image_path, size_mb = prepare_image(image_path)
    if verbose:
        print(f"Image size: {size_mb:.2f}MB")

    # Encode
    base64_image = encode_image(hq_image_path)

    # Optimized prompt focusing on left side
    prompt = """Look ONLY at the FAR LEFT SIDE of this sheet music where the key signatures are located.

CRITICAL: IGNORE any sharps/flats in the middle of the music - those are accidentals, NOT key signatures!

INSTRUCTIONS:
1. Count all staves (5 horizontal lines each)
2. For EACH staff, look at the area RIGHT AFTER the clef symbol (far left)
3. Count ONLY the sharps (#) or flats (♭) in that initial key signature area
4. Map to key name using the chart below

KEY SIGNATURE CHART:
- 0 sharps/flats = C
- 1 sharp = G
- 2 sharps = D
- 3 sharps = A
- 4 sharps = E
- 5 sharps = B
- 6 sharps = F#
- 1 flat = F
- 2 flats = Bb
- 3 flats = Eb
- 4 flats = Ab
- 5 flats = Db
- 6 flats = Gb

RESPONSE: Return ONLY a JSON array where each element is the key for one staff (top to bottom).

Example: ["G", "G", "G"] for 3 staves in G major

Now analyze the LEFT SIDE ONLY:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=800
        )

        # Clean up temp file
        if hq_image_path != image_path and os.path.exists(hq_image_path):
            os.remove(hq_image_path)

        # Parse response
        raw_response = response.choices[0].message.content

        if verbose:
            print(f"Response: {raw_response}")

        # Extract JSON array
        response_text = raw_response.strip()
        json_start = response_text.rfind('[')
        json_end = response_text.rfind(']') + 1

        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            json_text = json_text.replace('```json', '').replace('```', '').strip()
            key_signatures = json.loads(json_text)

            if isinstance(key_signatures, list):
                if verbose:
                    print(f"✓ Detected: {key_signatures}")
                return key_signatures
            else:
                if verbose:
                    print("✗ Error: Response not a list")
                return None
        else:
            if verbose:
                print("✗ Error: Could not find JSON array")
            return None

    except Exception as e:
        if hq_image_path != image_path and os.path.exists(hq_image_path):
            os.remove(hq_image_path)
        if verbose:
            print(f"✗ Error: {e}")
        return None


def detect_key_signature_single_measure(image_path, model="gpt-4o", verbose=True, show_image=True):
    """
    Detect key signature from a single first measure image (staff lines removed)

    Args:
        image_path: Path to first measure image
        model: GPT model to use (default: gpt-4o)
        verbose: Print progress messages
        show_image: Display the image being analyzed

    Returns:
        str: Key signature (e.g., "G", "C", "F") or None on error
    """
    if verbose:
        print(f"Analyzing: {Path(image_path).name}")

    # Display the image being analyzed
    if show_image:
        img = cv2.imread(image_path)
        if img is not None:
            # Add title to image
            h, w = img.shape[:2]
            display_img = np.ones((h + 40, w, 3), dtype=np.uint8) * 255
            display_img[40:, :] = img
            title = f"Processing: {Path(image_path).name}"
            cv2.putText(display_img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 0), 1, cv2.LINE_AA)

            # Create temporary preview file
            preview_path = Path(image_path).parent / "current_measure_preview.png"
            cv2.imwrite(str(preview_path), display_img)
            if verbose:
                print(f"  Preview saved: {preview_path}")

    # Prepare image
    hq_image_path, size_mb = prepare_image(image_path)
    if verbose:
        print(f"Image size: {size_mb:.2f}MB")

    # Encode
    base64_image = encode_image(hq_image_path)

    # Optimized prompt for single measure
    prompt = """Look at this single staff first measure image (staff lines removed).

CRITICAL: Look at the area RIGHT AFTER the clef symbol on the LEFT side.

INSTRUCTIONS:
1. Identify the clef symbol (treble or bass)
2. Count ONLY the sharps (#) or flats (♭) immediately after the clef (this is the key signature)
3. IGNORE any accidentals in the middle of the measure
4. Map to key name using the chart below

KEY SIGNATURE CHART:
- 0 sharps/flats = C
- 1 sharp = G
- 2 sharps = D
- 3 sharps = A
- 4 sharps = E
- 5 sharps = B
- 6 sharps = F#
- 1 flat = F
- 2 flats = Bb
- 3 flats = Eb
- 4 flats = Ab
- 5 flats = Db
- 6 flats = Gb

RESPONSE: Return ONLY the key signature letter(s) (e.g., "G" or "C" or "Bb"), nothing else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=800
        )

        # Clean up temp file
        if hq_image_path != image_path and os.path.exists(hq_image_path):
            os.remove(hq_image_path)

        # Parse response
        raw_response = response.choices[0].message.content.strip()

        if verbose:
            print(f"Response: {raw_response}")

        # Extract key signature (should be simple like "G" or "Bb")
        key_sig = raw_response.strip('"\'` \n')

        if key_sig:
            if verbose:
                print(f"✓ Detected key: {key_sig}")
            return key_sig
        else:
            if verbose:
                print("✗ Error: Empty response")
            return None

    except Exception as e:
        if hq_image_path != image_path and os.path.exists(hq_image_path):
            os.remove(hq_image_path)
        if verbose:
            print(f"✗ Error: {e}")
        return None


def run_orchestra_ai2(input_folder, output_folder=None):
    """
    Run Orchestra-AI-2 pipeline to extract first measures with staff lines removed

    Args:
        input_folder: Folder containing sheet music images to process
        output_folder: Where Orchestra-AI-2 will save results (defaults to Orchestra-AI-2/output)

    Returns:
        bool: True if successful, False otherwise
    """
    # Find Orchestra-AI-2 directory (search common locations)
    possible_paths = [
        Path("/Users/asherzaczepinski/Desktop/Orchestra-AI-2"),  # Common location
        Path(__file__).parent.parent / "Orchestra-AI-2",  # Relative to script
    ]

    orchestra_dir = None
    for path in possible_paths:
        if (path / "main.py").exists():
            orchestra_dir = path
            break

    if orchestra_dir is None:
        print(f"✗ Error: Orchestra-AI-2 directory not found")
        return False

    main_script = orchestra_dir / "main.py"

    # Default output folder
    if output_folder is None:
        output_folder = str(orchestra_dir / "output")

    if not main_script.exists():
        print(f"✗ Error: Orchestra-AI-2 main.py not found at {main_script}")
        return False

    # Convert paths to absolute paths
    input_folder = str(Path(input_folder).resolve())
    output_folder = str(Path(output_folder).resolve())

    print(f"\n{'='*60}")
    print("RUNNING ORCHESTRA-AI-2 PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {input_folder}")
    print(f"Output: {output_folder}")
    print(f"{'='*60}\n")

    try:
        # Run Orchestra-AI-2 main.py with the input and output folders
        result = subprocess.run(
            [sys.executable, str(main_script), input_folder, output_folder],
            cwd=str(orchestra_dir),
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        print(f"\n✓ Orchestra-AI-2 pipeline completed successfully!\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Orchestra-AI-2 pipeline failed with exit code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Error running Orchestra-AI-2: {e}")
        return False


def create_first_measures_visualization(image_files, results, output_folder):
    """
    Create a visualization showing all first measures with their detected key signatures

    Args:
        image_files: List of Path objects for first measure images
        results: List of detection results with staff numbers and key signatures
        output_folder: Where to save the visualization
    """
    if not image_files or not results:
        return

    # Create a mapping of staff number to key signature
    staff_to_key = {r['staff_number']: r['key_signature'] for r in results if r['success']}

    # Load all images
    images = []
    max_width = 0
    total_height = 0

    for img_file in sorted(image_files):
        # Extract staff number from filename
        staff_num = int(img_file.stem.split('_')[1])

        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Get key signature for this staff
        key_sig = staff_to_key.get(staff_num, "Unknown")

        # Add text label to image
        label_height = 40
        labeled_img = np.ones((img.shape[0] + label_height, img.shape[1], 3), dtype=np.uint8) * 255
        labeled_img[label_height:, :] = img

        # Add text
        text = f"Staff {staff_num}: {key_sig} major"
        cv2.putText(labeled_img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

        images.append(labeled_img)
        max_width = max(max_width, labeled_img.shape[1])
        total_height += labeled_img.shape[0] + 10  # 10px spacing

    # Create composite image
    composite = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255

    current_y = 0
    for img in images:
        h = img.shape[0]
        composite[current_y:current_y + h, :img.shape[1]] = img
        current_y += h + 10

    # Save visualization
    vis_path = Path(output_folder) / f"first_measures_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(vis_path), composite)
    print(f"✓ Visualization saved to: {vis_path}")


def process_first_measures(folder_path=None, output_folder="results", run_pipeline=False, input_folder=None):
    """
    Process first measure images with staff lines removed and detect key signatures

    Args:
        folder_path: Path to folder containing first measure images (staff lines removed)
        output_folder: Where to save key signature detection results
        run_pipeline: If True, run Orchestra-AI-2 pipeline first to generate first measures
        input_folder: Input folder for Orchestra-AI-2 (required if run_pipeline=True)
    """

    # Step 1: Run Orchestra-AI-2 pipeline if requested
    if run_pipeline:
        if not input_folder:
            print("✗ Error: input_folder must be specified when run_pipeline=True")
            return

        success = run_orchestra_ai2(input_folder)
        if not success:
            print("✗ Failed to run Orchestra-AI-2 pipeline")
            return

    # Step 2: Process the generated first measures
    # Default folder path if not specified
    if folder_path is None:
        # Search for Orchestra-AI-2 output directory
        possible_paths = [
            Path("/Users/asherzaczepinski/Desktop/Orchestra-AI-2/output/first_measures_staff_removed"),
            Path(__file__).parent.parent / "Orchestra-AI-2/output/first_measures_staff_removed",
        ]
        for path in possible_paths:
            if path.exists():
                folder_path = str(path)
                break

        if folder_path is None:
            print("✗ Error: Could not find first_measures_staff_removed directory")
            return

    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(exist_ok=True)

    # Find all first measure images
    image_files = sorted(folder.glob("staff_*_first_measure_no_lines.png"))

    if not image_files:
        print(f"No first measure images found in {folder}/")
        return

    print(f"\n{'='*60}")
    print("GPT-4O KEY SIGNATURE DETECTION")
    print(f"{'='*60}")
    print(f"Processing {len(image_files)} first measure(s)...")
    print("=" * 60)

    results = []
    for img_file in image_files:
        # Extract staff number from filename
        staff_num = img_file.stem.split('_')[1]

        # Detect key signature for this measure (show each image as it's processed)
        key_sig = detect_key_signature_single_measure(str(img_file), verbose=True, show_image=True)

        results.append({
            'staff_number': int(staff_num),
            'filename': img_file.name,
            'success': key_sig is not None,
            'key_signature': key_sig,
            'timestamp': datetime.now().isoformat()
        })
        print("-" * 60)

    # Save results
    output_file = output / f"key_signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}\n")

    # Create visualization of all first measures with key signatures
    print("Creating visualization...")
    create_first_measures_visualization(image_files, results, output)

    # Summary
    print("SUMMARY:")
    for result in sorted(results, key=lambda x: x['staff_number']):
        if result['success']:
            print(f"  ✓ Staff {result['staff_number']}: {result['key_signature']}")
        else:
            print(f"  ✗ Staff {result['staff_number']}: FAILED")


def batch_process(folder_path="test_images", output_folder="results"):
    """Process all images in folder and save results"""

    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {folder}/")
        return

    print(f"Processing {len(image_files)} image(s)...")
    print("=" * 60)

    results = []
    for img_file in image_files:
        key_sigs = detect_key_signatures(str(img_file), verbose=True)

        results.append({
            'filename': img_file.name,
            'success': key_sigs is not None,
            'key_signatures': key_sigs,
            'staff_count': len(key_sigs) if key_sigs else 0,
            'timestamp': datetime.now().isoformat()
        })
        print("-" * 60)

    # Save results
    output_file = output / f"key_signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}\n")

    # Summary
    print("SUMMARY:")
    for result in results:
        if result['success']:
            print(f"  ✓ {result['filename']}: {result['key_signatures']}")
        else:
            print(f"  ✗ {result['filename']}: FAILED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect key signatures from sheet music using Orchestra-AI-2 + GPT-4o'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input folder containing sheet music images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output folder for results JSON'
    )
    parser.add_argument(
        '--skip-pipeline',
        action='store_true',
        help='Skip Orchestra-AI-2 and only process existing first measures'
    )

    args = parser.parse_args()

    if args.skip_pipeline:
        # Just process existing first measures
        print("Processing existing first measures (skipping Orchestra-AI-2 pipeline)\n")
        process_first_measures(
            output_folder=args.output,
            run_pipeline=False
        )
    else:
        # Run full pipeline (default behavior)
        print("Running full pipeline: Orchestra-AI-2 → GPT-4o key signature detection\n")
        process_first_measures(
            output_folder=args.output,
            run_pipeline=True,
            input_folder=args.input
        )
