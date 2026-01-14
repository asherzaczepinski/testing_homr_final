# GPT-4 Vision Key Signature Detection - Test Results

## Test Date: January 13, 2026

## Summary: ⚠️ NOT RECOMMENDED

After testing GPT-4 Vision for key signature detection on sheet music, **the results show significant accuracy issues**.

---

## Test Results

### Test Image: `2026-01-08 20.png` (FALCON FANFARE - Bb Clarinet 1)

**Actual (Ground Truth):**
- Number of staves: **10**
- Key signature: **G major (1 sharp: F#)**
- All staves have the same key signature

### Attempt 1: Basic Prompt
**GPT-4o Response:**
```json
["C", "C", "C", "C", "C", "C", "C", "C", "C", "C"]
```
- ✓ Correct staff count: 10
- ❌ Wrong key: Said "C" (no sharps/flats), actually "G" (1 sharp)
- **Accuracy: 0/10 staves correct**

### Attempt 2: Improved Prompt with Detailed Instructions
**GPT-4o Response:**
```json
["F"]
```
- ❌ Wrong staff count: Said 1, actually 10
- ❌ Wrong key: Said "F" (1 flat), actually "G" (1 sharp)
- ❌ Confused sharps with flats
- **Accuracy: 0% correct**

---

## Key Findings

### Critical Issues Identified:

1. **Cannot distinguish sharps from flats**
   - Mistook a sharp (#) for a flat (♭)
   - This is a fundamental failure for music notation

2. **Staff counting is unreliable**
   - First attempt: Counted 10 (correct)
   - Second attempt: Counted 1 (incorrect)
   - Inconsistent results

3. **Key signature misidentification**
   - Neither attempt correctly identified G major
   - Confused with C major and F major

4. **Inconsistent behavior**
   - More detailed prompting actually made results worse
   - No clear pattern to the errors

---

## Why GPT-4 Vision Struggles

1. **Fine-grained symbol recognition**: Musical notation requires precise identification of small symbols (sharps vs flats)
2. **Spatial reasoning**: Needs to correlate symbols with their position on staff lines
3. **Not trained for music**: GPT-4 Vision is general-purpose, not specialized for music notation
4. **Image resolution**: Key signature symbols can be small and detailed

---

## Recommendations

### ❌ DO NOT USE: GPT-4 Vision (Current Approach)
- Unreliable accuracy
- Inconsistent results
- Cannot be trusted for production use

### ✅ BETTER ALTERNATIVES:

#### 1. **Specialized OMR (Optical Music Recognition) Tools**
   - **Audiveris**: Open-source OMR that can extract key signatures
   - **music21 + OMR**: Python library for music analysis
   - **Pros**: Built specifically for music notation
   - **Cons**: May require training/configuration

#### 2. **Computer Vision + Rules-Based Approach**
   - Use OpenCV to detect key signature region
   - Template matching for sharp/flat symbols
   - Count symbols and map to key
   - **Pros**: More reliable, deterministic
   - **Cons**: Requires development effort

#### 3. **Your Existing HOMR Pipeline**
   - HOMR already converts to MusicXML/semantic tokens
   - MusicXML contains key signature information
   - Extract key signatures from HOMR's output
   - **Pros**: Already integrated, accurate
   - **Cons**: Requires HOMR processing first

#### 4. **Fine-tuned Vision Model**
   - Train a specialized model on music notation
   - Use a dataset of labeled key signatures
   - **Pros**: Could be very accurate
   - **Cons**: Requires ML expertise and training data

---

## Recommended Next Step: Use HOMR Output

Since you already have HOMR in your pipeline, **the best approach is to extract key signatures from HOMR's output** rather than using GPT-4 Vision.

HOMR outputs include:
- Semantic tokens with key signature information
- MusicXML (if converted) with `<key>` elements

This would be:
- ✅ More accurate (HOMR is trained for music notation)
- ✅ Already integrated
- ✅ No additional API costs
- ✅ Reliable and consistent

---

## Cost Analysis

**GPT-4 Vision Costs (if you were to use it):**
- ~$0.01 - $0.05 per image
- For 1000 images: $10 - $50
- **Plus the cost of inaccuracy!**

**HOMR Approach:**
- $0 (already running)
- More accurate
- Better overall value

---

## Conclusion

While GPT-4 Vision is powerful for many visual tasks, **it is not suitable for precise music notation analysis** like key signature detection. The errors are too frequent and inconsistent to be reliable.

**Action Items:**
1. ❌ Do not integrate GPT-4 Vision for key signatures
2. ✅ Use HOMR's output to extract key signature data
3. ✅ Parse MusicXML or semantic tokens for key information
4. ✅ Keep the testinggpt folder for future experiments

---

## Test Environment Preserved

The `testinggpt/` folder has been set up and is ready if you want to:
- Test GPT-4 on other music analysis tasks (lyrics, structure, etc.)
- Experiment with different prompting strategies
- Try newer models when they're released
- Use for non-notation tasks (image descriptions, etc.)

But for **key signature detection specifically**: Use HOMR's output instead.
