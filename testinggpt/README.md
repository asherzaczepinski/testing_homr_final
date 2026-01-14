# GPT-4 Vision Key Signature Detection

This folder contains a testing environment for using **GPT-4 Vision** to automatically detect key signatures from sheet music images.

## What It Does

This tool:
- Takes sheet music images as input
- Uses OpenAI's GPT-4 Vision API to analyze the image
- Identifies the key signature for each staff line
- Returns results as an array: `["F#", "Bb", "C", ...]`
- Each index represents one staff, starting from 0 (top staff)

## Folder Structure

```
testinggpt/
├── test_images/           # Put your sheet music images HERE
├── results/              # JSON results saved here
├── venv/                 # Python virtual environment
├── .env                  # API key (DO NOT commit to git!)
├── .gitignore           # Protects your API key
├── test_gpt_key_detection.py   # Main test script
└── README.md            # This file
```

## Quick Start

### 1. Setup (Already Done!)
- ✓ Virtual environment created
- ✓ OpenAI library installed
- ✓ API key configured in `.env`
- ✓ Input image copied to `test_images/`

### 2. Run the Test

```bash
cd testinggpt
source venv/bin/activate
python test_gpt_key_detection.py
```

### 3. Check Results
- Console output shows key signatures immediately
- Detailed results saved to `results/key_detection_results_TIMESTAMP.json`

## Example Output

```
Found 1 image(s) to process
============================================================

Analyzing: sheet_music.png
Using model: gpt-4o

Raw response:
["Bb", "Bb"]

✓ Key signatures: ["Bb", "Bb"]
  Staff count: 2
------------------------------------------------------------

============================================================
Processing complete!
Results saved to: results/key_detection_results_20260113_215300.json
============================================================

SUMMARY:
  sheet_music.png: ["Bb", "Bb"]
```

## Understanding the Output

The script returns an array where:
- **Index 0** = First (top) staff
- **Index 1** = Second staff
- **Index 2** = Third staff
- And so on...

### Key Signature Notation
- **C** = No sharps or flats (C major / A minor)
- **G** = 1 sharp (G major / E minor)
- **D** = 2 sharps (D major / B minor)
- **F** = 1 flat (F major / D minor)
- **Bb** = 2 flats (Bb major / G minor)
- **F#** = 6 sharps (F# major / D# minor)
- etc.

## Results Format

Results are saved as JSON with detailed information:

```json
[
  {
    "success": true,
    "key_signatures": ["Bb", "Bb"],
    "raw_response": "[\"Bb\", \"Bb\"]",
    "staff_count": 2,
    "filename": "sheet_music.png",
    "timestamp": "2026-01-13T21:53:00.123456"
  }
]
```

## API Costs

This uses OpenAI's GPT-4o model with vision:
- **Cost**: ~$0.01 - $0.05 per image (varies by image size)
- **Speed**: 2-5 seconds per image
- Monitor usage at: https://platform.openai.com/usage

## Python API Example

You can also use the functions directly in your code:

```python
from test_gpt_key_detection import analyze_key_signatures

# Analyze a single image
result = analyze_key_signatures("path/to/image.png")

if result['success']:
    key_sigs = result['key_signatures']
    print(f"Key signatures: {key_sigs}")
    print(f"First staff: {key_sigs[0]}")
    print(f"Second staff: {key_sigs[1]}")
else:
    print(f"Error: {result['error']}")
```

## Customization

### Use a Different Model
Edit the script to use different GPT models:

```python
# In test_gpt_key_detection.py, line ~50
result = analyze_key_signatures(image_path, model="gpt-4o-mini")  # Cheaper, faster
```

Available models:
- `gpt-4o` - Best quality (default)
- `gpt-4o-mini` - Faster and cheaper
- `gpt-4-turbo` - Previous generation

### Batch Processing
The script automatically processes all images in `test_images/`:

```bash
# Add multiple images
cp /path/to/sheets/*.png test_images/

# Run batch processing
python test_gpt_key_detection.py
```

## Troubleshooting

### API Key Issues
```bash
# Check if .env file exists and has your key
cat .env

# Make sure the virtual environment is activated
source venv/bin/activate
```

### No Images Found
Make sure images are in the `test_images/` folder with supported extensions:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

### JSON Parse Errors
If GPT returns text instead of JSON, the script will show the raw response. This can happen with:
- Very complex or unclear images
- Images with no visible staff lines
- Corrupted images

### Rate Limits
If you hit rate limits:
- Wait a few seconds between requests
- Upgrade your OpenAI plan
- Use `gpt-4o-mini` instead

## Security Note

⚠️ **IMPORTANT**: The `.env` file contains your API key and is excluded from git via `.gitignore`. Never commit or share this file!

## Next Steps

1. Test with more sheet music images
2. Evaluate accuracy of key signature detection
3. Compare results with actual key signatures
4. If accurate, integrate into your main pipeline

## More Information

- OpenAI Vision API: https://platform.openai.com/docs/guides/vision
- API Pricing: https://openai.com/api/pricing/
- Key Signatures Reference: https://en.wikipedia.org/wiki/Key_signature
