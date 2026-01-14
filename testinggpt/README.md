# GPT-4o Key Signature Detection

Production-ready tool for detecting key signatures from sheet music images using GPT-4o Vision API.

## What It Does

Analyzes sheet music images and returns key signatures for each staff as an array:
```python
["G", "G", "G", "Bb", "F"]  # Each index = one staff
```

## Quick Start

### 1. Setup (Already Done!)
- ✓ Virtual environment created
- ✓ OpenAI library installed
- ✓ API key configured

### 2. Run Detection

**Batch Processing (all images in test_images/):**
```bash
cd testinggpt
source venv/bin/activate
python detect_key_signatures.py
```

**Single Image (Python API):**
```python
from detect_key_signatures import detect_key_signatures

keys = detect_key_signatures("path/to/sheet_music.png")
print(keys)  # ['G', 'G', 'G', ...]
```

### 3. Check Results
- Console output shows immediate results
- JSON results saved to `results/` folder

## Example Output

```
Processing 1 image(s)...
Analyzing: sheet_music.png
Image size: 1.98MB
Response: ["G", "G", "G"]
✓ Detected: ['G', 'G', 'G']

SUMMARY:
  ✓ sheet_music.png: ['G', 'G', 'G']
```

## Key Signature Format

- **C** = No sharps/flats
- **G** = 1 sharp
- **D** = 2 sharps
- **A** = 3 sharps
- **F** = 1 flat
- **Bb** = 2 flats
- **Eb** = 3 flats
- etc.

## Accuracy

Based on testing:
- ✅ **Key signature detection: Accurate** (correctly identifies G major, F major, etc.)
- ⚠️ **Staff counting: ~90% accurate** (may be off by 1-2 staves)
- ✅ **Sharp vs Flat: Reliable** when using the "focus left" approach

See `FINAL_RESULTS.md` for detailed test results.

## Cost

- **~$0.01-0.02 per image**
- GPT-4o pricing: $2.50/1M input tokens, $10/1M output tokens
- Processing time: 2-3 seconds per image

## API Limits

- **Image size:** 20MB max
- **Rate limit:** 500 requests/min (default)
- **Formats:** PNG, JPG, JPEG, BMP

## Files

```
testinggpt/
├── detect_key_signatures.py   # Main script (USE THIS)
├── test_images/               # Put images here
├── results/                   # JSON output saved here
├── FINAL_RESULTS.md          # Detailed test results
├── .env                       # API key (DO NOT COMMIT)
└── venv/                     # Python environment
```

## Python API

```python
from detect_key_signatures import detect_key_signatures

# Detect with verbose output
keys = detect_key_signatures("image.png", verbose=True)

# Detect quietly
keys = detect_key_signatures("image.png", verbose=False)

# Use different model
keys = detect_key_signatures("image.png", model="gpt-4o-mini")  # Cheaper

# Process results
if keys:
    for i, key in enumerate(keys):
        print(f"Staff {i+1}: {key}")
```

## Troubleshooting

**Module not found errors:**
```bash
source venv/bin/activate
```

**API key errors:**
Check that `.env` file exists and contains your OpenAI API key.

**No images found:**
Add images to the `test_images/` folder.

**JSON parse errors:**
The model returned unexpected format - try rerunning or check image quality.

## Integration

To add this to your pipeline:

```python
import sys
sys.path.append('/path/to/testinggpt')

from detect_key_signatures import detect_key_signatures

# In your pipeline
image_path = "path/to/sheet_music.png"
key_signatures = detect_key_signatures(image_path, verbose=False)

if key_signatures:
    print(f"Detected keys: {key_signatures}")
    # Use the key signatures in your processing...
```

## Security

⚠️ **IMPORTANT:** The `.env` file contains your API key and is git-ignored. Never commit or share this file!

## More Info

- OpenAI API: https://platform.openai.com/docs
- Pricing: https://openai.com/api/pricing/
- Test Results: See `FINAL_RESULTS.md`
