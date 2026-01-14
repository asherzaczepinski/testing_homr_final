#!/usr/bin/env python3
"""
PRODUCTION VERSION: GPT-4o Key Signature Detection
Focuses on left side key signature area for accurate detection
TESTED: Correctly identifies G major (1 sharp)
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

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
    batch_process()
