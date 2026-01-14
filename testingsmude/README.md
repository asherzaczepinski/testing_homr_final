# Smude Testing Environment

This folder contains a testing environment for **smude** - a sheet music dewarping tool that straightens curved pages from smartphone photos.

## What is Smude?

Smude processes smartphone photographs of sheet music by:
- Extracting the music page from the photo
- Straightening curved/warped pages (like from bound books)
- Converting to clean binary (black and white) images
- Using deep learning to detect and correct staff lines

## Folder Structure

```
testingsmude/
├── smude/              # Cloned smude repository
├── test_images/        # Put your sheet music photos HERE
├── output_images/      # Processed results will appear here
├── venv/              # Python virtual environment
├── test_smude.py      # Main test script
└── README.md          # This file
```

## Quick Start

### 1. Add Your Images
Place your sheet music photos (JPG, PNG, etc.) in the `test_images/` folder.

### 2. Run the Test Script

Activate the virtual environment and run the script:

```bash
cd testingsmude
source venv/bin/activate
python test_smude.py
```

### 3. Check Results
Find the processed images in the `output_images/` folder with filenames like `dewarped_<original_name>.png`

## Using Smude Directly (Command Line)

You can also use smude's command-line interface for individual images:

```bash
source venv/bin/activate
smude test_images/my_image.jpg -o output_images/result.png
```

Options:
- `-o, --outfile`: Specify output file (default: result.png)
- `--no-binarization`: Skip binarization (keep grayscale)
- `--use-gpu`: Use GPU for faster processing

## Tips for Best Results

- **Include full page**: Capture the entire page plus some margins
- **Even lighting**: Avoid shadows and uneven lighting
- **Sharp focus**: Blurry images will not work well
- **Cylindrical curves**: Works best with curved pages from bound books

## First Run Note

The first time you run smude, it will download a ~348 MB deep learning model. This is normal and only happens once.

## Troubleshooting

### Virtual Environment Not Activated
If you get "command not found: python" or import errors:
```bash
source venv/bin/activate
```

### No Images Found
Make sure you've placed images in the `test_images/` folder.

### Processing Errors
- Check that images are clear and in focus
- Ensure the full page is visible in the photo
- Try different lighting conditions

## Python API Example

You can also use smude in your own Python scripts:

```python
from skimage.io import imread, imsave
from smude import Smude

# Load image
image = imread("test_images/sheet_music.jpg")

# Initialize Smude
smude = Smude(use_gpu=False, binarize_output=True)

# Process
result = smude.process(image)

# Save
imsave("output_images/result.png", result)
```

## More Information

- GitHub: https://github.com/sonovice/smude
- See `smude/README.md` for detailed documentation
- See `smude/example.py` for more code examples
