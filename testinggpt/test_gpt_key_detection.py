#!/usr/bin/env python3
"""
GPT-4 Vision Key Signature Detection
Analyzes sheet music images and extracts key signatures for each staff
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def encode_image(image_path):
    """Encode image to base64 for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_key_signatures(image_path, model="gpt-5.2"):
    """
    Analyze sheet music image and extract key signatures for each staff

    Args:
        image_path: Path to the sheet music image
        model: GPT model to use (default: gpt-4o)

    Returns:
        dict with 'key_signatures' array and raw response
    """
    print(f"\nAnalyzing: {Path(image_path).name}")
    print(f"Using model: {model}")

    # Encode image
    base64_image = encode_image(image_path)

    # Create the prompt
    prompt = """Analyze this sheet music image and identify the key signature for each staff.

IMPORTANT INSTRUCTIONS:
1. Look at each staff line (horizontal line with 5 lines where notes are written)
2. For each staff, identify the key signature by looking at the sharps (#) or flats (♭) at the beginning of the staff, right after the clef
3. Return ONLY a valid JSON array where each element represents one staff (starting from index 0 at the top)
4. Use standard key signature notation: "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"
5. If a staff has no sharps or flats in the key signature, use "C" (C major/A minor)
6. Count staves from top to bottom, starting at index 0

RESPONSE FORMAT (JSON only, no other text):
["key1", "key2", "key3", ...]

Example responses:
- Single staff in G major: ["G"]
- Two staves in F major: ["F", "F"]
- Three different keys: ["D", "Bb", "C"]

Now analyze this image and return ONLY the JSON array:"""

    try:
        # Call GPT-4 Vision API
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
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Extract response
        raw_response = response.choices[0].message.content
        print(f"\nRaw response:\n{raw_response}")

        # Try to parse JSON from response
        # Sometimes GPT adds markdown code blocks, so we need to clean it
        response_text = raw_response.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # Parse JSON
        key_signatures = json.loads(response_text)

        if not isinstance(key_signatures, list):
            raise ValueError("Response is not a list/array")

        return {
            'success': True,
            'key_signatures': key_signatures,
            'raw_response': raw_response,
            'staff_count': len(key_signatures)
        }

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {
            'success': False,
            'error': f'JSON parse error: {e}',
            'raw_response': raw_response
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def process_images_in_folder(folder_path="test_images", output_folder="results"):
    """Process all images in the specified folder"""

    folder = Path(folder_path)
    output = Path(output_folder)
    output.mkdir(exist_ok=True)

    if not folder.exists():
        print(f"Error: {folder} does not exist!")
        return

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in folder.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {folder}/")
        print("Please add sheet music images to the test_images/ folder")
        return

    print(f"Found {len(image_files)} image(s) to process")
    print("=" * 60)

    # Process each image
    results = []
    for img_file in image_files:
        result = analyze_key_signatures(str(img_file))
        result['filename'] = img_file.name
        result['timestamp'] = datetime.now().isoformat()
        results.append(result)

        if result['success']:
            print(f"✓ Key signatures: {result['key_signatures']}")
            print(f"  Staff count: {result['staff_count']}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")

        print("-" * 60)

    # Save results to JSON file
    output_file = output / f"key_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 60}")

    # Print summary
    print("\nSUMMARY:")
    for result in results:
        if result['success']:
            print(f"  {result['filename']}: {result['key_signatures']}")
        else:
            print(f"  {result['filename']}: FAILED")


if __name__ == "__main__":
    process_images_in_folder()
