#!/usr/bin/env python3
"""
Improved GPT-4 Vision Key Signature Detection with better prompting
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_key_signatures_improved(image_path, model="gpt-5.2"):
    """Improved key signature detection with more detailed prompting"""

    print(f"\nAnalyzing: {Path(image_path).name}")
    print(f"Using model: {model}")

    base64_image = encode_image(image_path)

    # Improved prompt with more specific instructions
    prompt = """You are a music notation expert. Analyze this sheet music image carefully and identify the key signature for each staff.

STEP-BY-STEP INSTRUCTIONS:
1. Locate each staff (the 5 horizontal lines where musical notes are placed)
2. For each staff, look immediately after the clef symbol (treble clef 🎼 or bass clef)
3. Count the number and type of accidentals in the key signature:
   - Sharp symbols (#) that appear right after the clef
   - Flat symbols (♭) that appear right after the clef
   - These are NOT individual note accidentals - they are the key signature symbols at the start
4. Convert the key signature to the major key name based on this chart:

SHARP KEYS:
- 0 sharps = C
- 1 sharp (F#) = G
- 2 sharps (F#, C#) = D
- 3 sharps (F#, C#, G#) = A
- 4 sharps (F#, C#, G#, D#) = E
- 5 sharps (F#, C#, G#, D#, A#) = B
- 6 sharps (F#, C#, G#, D#, A#, E#) = F#
- 7 sharps = C#

FLAT KEYS:
- 1 flat (Bb) = F
- 2 flats (Bb, Eb) = Bb
- 3 flats (Bb, Eb, Ab) = Eb
- 4 flats (Bb, Eb, Ab, Db) = Ab
- 5 flats (Bb, Eb, Ab, Db, Gb) = Db
- 6 flats (Bb, Eb, Ab, Db, Gb, Cb) = Gb
- 7 flats = Cb

5. Count staves from top to bottom, starting at index 0

RESPONSE FORMAT:
First, explain what you see in each staff's key signature area (for verification).
Then provide ONLY a JSON array with the key names.

Example response format:
Analysis: I can see 2 staves. The first staff has one sharp (F#) after the treble clef, indicating G major. The second staff also has one sharp (F#), also indicating G major.

["G", "G"]

Now analyze this image:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        raw_response = response.choices[0].message.content
        print(f"\nRaw response:\n{raw_response}")
        print("\n" + "=" * 60)

        # Extract JSON from response
        response_text = raw_response.strip()

        # Find the JSON array in the response
        json_start = response_text.rfind('[')
        json_end = response_text.rfind(']') + 1

        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]

            # Remove markdown if present
            json_text = json_text.replace('```json', '').replace('```', '').strip()

            key_signatures = json.loads(json_text)

            if not isinstance(key_signatures, list):
                raise ValueError("Response is not a list/array")

            return {
                'success': True,
                'key_signatures': key_signatures,
                'raw_response': raw_response,
                'staff_count': len(key_signatures)
            }
        else:
            raise ValueError("Could not find JSON array in response")

    except Exception as e:
        print(f"Error: {e}")
        return {
            'success': False,
            'error': str(e),
            'raw_response': raw_response if 'raw_response' in locals() else None
        }


def process_images(folder_path="test_images", output_folder="results"):
    """Process all images"""

    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {folder}/")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print("=" * 60)

    results = []
    for img_file in image_files:
        result = analyze_key_signatures_improved(str(img_file))
        result['filename'] = img_file.name
        result['timestamp'] = datetime.now().isoformat()
        results.append(result)

        if result['success']:
            print(f"✓ Key signatures: {result['key_signatures']}")
            print(f"  Staff count: {result['staff_count']}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")

        print("-" * 60)

    # Save results
    output_file = output / f"improved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete! Results saved to: {output_file}\n")

    print("SUMMARY:")
    for result in results:
        if result['success']:
            print(f"  {result['filename']}: {result['key_signatures']}")
        else:
            print(f"  {result['filename']}: FAILED")


if __name__ == "__main__":
    process_images()
