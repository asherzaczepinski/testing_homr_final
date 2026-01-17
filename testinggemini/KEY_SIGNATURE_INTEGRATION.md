# Key Signature Detection & Integration with HOMR

## Overview

This system uses Gemini Flash 2.0 to detect key signatures and integrates them with HOMR's existing circle of fifths system to properly handle sharps and flats throughout the score.

## How It Works

### 1. Key Signature Detection (Gemini)

The `homr_key_detection_gemini.py` script:

1. **Detects staffs** using HOMR
2. **Extracts first measures** from each staff
3. **Isolates key signature region** (area right after the clef)
4. **Sends to Gemini Flash 2.0** which returns:
   - Count of sharps or flats
   - Type (sharp or flat)
5. **Infers the musical key** using circle of fifths mapping:

```
Sharps:                  Flats:
0 = CM (C Major)        0 = CM (C Major)
1 = GM (G Major)        1 = FM (F Major)
2 = DM (D Major)        2 = BbM (Bb Major)
3 = AM (A Major)        3 = EbM (Eb Major)
4 = EM (E Major)        4 = AbM (Ab Major)
5 = BM (B Major)        5 = DbM (Db Major)
6 = F#M (F# Major)      6 = GbM (Gb Major)
7 = C#M (C# Major)      7 = CbM (Cb Major)
```

6. **Applies key signature** to the staff and all its notes via `circle_of_fifth` value

### 2. Integration with HOMR's Circle of Fifths System

HOMR has a built-in `KeyTransformation` class (`homr/circle_of_fifths.py`) that handles:

#### Key Signature Application
- When `circle_of_fifth = 1` (G Major), all F notes become F#
- When `circle_of_fifth = -2` (Bb Major), all B and E notes become Bb and Eb
- The transformation applies to **all octaves** of affected notes

#### Measure-Level Accidental Overrides

The `KeyTransformation` class handles accidentals that override the key signature:

```python
# Example: G Major (1 sharp: F#)
key = KeyTransformation(1)  # F notes are sharp by default

# If a natural sign appears on F4:
key.add_accidental("F4", "N")  # Removes F from sharps set
# → F4 becomes natural for the rest of the measure

# At the next bar line:
key = key.reset_at_end_of_measure()  # Resets to G Major
# → F notes are sharp again
```

**Key Rules:**
1. **Key signature applies to entire staff** - All instances of affected notes across all octaves
2. **Accidentals override within measure** - Sharp, flat, or natural signs override the key signature
3. **Overrides persist until bar line** - An accidental affects all subsequent instances of that note in the measure
4. **Reset at measure boundaries** - Each new measure starts fresh with the key signature
5. **Each staff has independent key** - Different staffs can have different keys (though typically they're the same)

### 3. Code Flow

```
Image → HOMR Detection → Extract Key Signature Region → Gemini Analysis
                                                              ↓
                                              (count=1, type="sharp")
                                                              ↓
                                              Infer Key: GM, CoF=1
                                                              ↓
                    Apply to Staff: staff.circle_of_fifth = 1
                                    note.circle_of_fifth = 1
                                                              ↓
                    HOMR's KeyTransformation automatically:
                    - Makes all F notes sharp
                    - Handles accidental overrides per measure
                    - Resets at each bar line
```

## Files Modified

1. **testinggemini/homr_key_detection_gemini.py**
   - Added key inference logic
   - Applies circle_of_fifth to each staff
   - Shows inferred key in output

2. **homr_repo/homr/debug.py**
   - Modified `write_model_input_image()` to NOT save tesseract files when debug=False
   - Prevents clutter from OCR input images

## Example Output

```
Staff 1: 1 sharp(s) → Key: GM (CoF: 1)
  ✓ Applied key signature GM to staff 1

  This means:
  - All F notes in this staff are F# (unless overridden by natural/flat)
  - Overrides only apply within their measure
  - Next measure resets to F# again
```

## Testing

Run on your input folder:
```bash
venv/bin/python testinggemini/homr_key_detection_gemini.py "inputfolder/2026-01-08 20.png" --output testinggemini/results
```

Results include:
- Key signature images for each staff
- JSON file with detected keys and circle of fifths values
- First measure visualizations with clef and key signature regions marked

## Future Integration

To fully integrate with HOMR's note detection:

1. The `circle_of_fifth` values are already set on Staff and Note objects
2. HOMR's `maintain_accidentals_during_measure()` function uses `KeyTransformation`
3. When processing notes, HOMR will automatically:
   - Apply the key signature to determine base pitch
   - Track accidental overrides within measures
   - Reset at bar lines

The infrastructure is in place - key signatures detected by Gemini now feed directly into HOMR's existing accidental handling system!
