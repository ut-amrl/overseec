# Image Resizer - Match Target Dimensions

Resize a "method" image to exactly match a "target" image's dimensions, with intelligent handling of aspect ratio differences through padding or cropping.

## Features

- ✅ Resizes images to exact target dimensions
- ✅ Preserves aspect ratio during scaling
- ✅ Automatically pads or crops to reach exact dimensions
- ✅ Configurable padding color (black, white, transparent, custom)
- ✅ Multiple crop modes (center, top, bottom, left, right)
- ✅ High-quality LANCZOS resampling
- ✅ Supports all common image formats (PNG, JPEG, TIFF, BMP, etc.)
- ✅ Original files remain unchanged

## Installation

### Requirements

- Python 3.7+
- Pillow (PIL)
- NumPy

### Setup

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install Pillow numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Syntax

```bash
python image_resizer.py <target_image> <method_image> [options]
```

### Common Examples

#### 1. Basic Resize (Default Settings)
```bash
python image_resizer.py target.png method.png
```
- Resizes `method.png` to match `target.png` dimensions
- Centers the scaled image
- Pads with black or crops from center as needed
- Saves as `method_resized.png`

#### 2. Custom Output Filename
```bash
python image_resizer.py target.png method.png -o output.png
```

#### 3. White Padding
```bash
python image_resizer.py target.png method.png --pad-color 255,255,255
```

#### 4. Transparent Padding (for PNG/RGBA)
```bash
python image_resizer.py target.png method.png --pad-color none
```

#### 5. Custom Padding Color (e.g., red)
```bash
python image_resizer.py target.png method.png --pad-color 255,0,0
```

#### 6. Crop from Top
```bash
python image_resizer.py target.png method.png --crop-mode top
```

#### 7. Complete Example
```bash
python image_resizer.py \
  target_1920x1080.png \
  method_photo.jpg \
  -o resized_photo.jpg \
  --pad-color 128,128,128 \
  --crop-mode center
```

## How It Works

### Step 1: Aspect Ratio Preservation

The program first scales the method image while preserving its aspect ratio:

- **If method is wider**: Scale to match target width
- **If method is taller**: Scale to match target height

### Step 2: Padding or Cropping

After scaling, the image is adjusted to exact target dimensions:

**Padding (when scaled image is smaller):**
- Adds padding around the image
- Default: Black (0,0,0) for RGB, Transparent (0,0,0,0) for RGBA
- Configurable with `--pad-color`

**Cropping (when scaled image is larger):**
- Removes excess pixels
- Default: Center crop
- Configurable with `--crop-mode`

### Example Scenarios

#### Scenario 1: Wider Method Image
```
Target: 800x600
Method: 1600x800

1. Scale to match width: 800x400
2. Pad height: Add 100px top and 100px bottom
Result: 800x600 ✓
```

#### Scenario 2: Taller Method Image
```
Target: 800x600
Method: 400x800

1. Scale to match height: 300x600
2. Pad width: Add 250px left and 250px right
Result: 800x600 ✓
```

#### Scenario 3: Much Wider Method
```
Target: 800x600
Method: 2000x400

1. Scale to match width: 800x160
2. Pad height: Add 220px top and 220px bottom
Result: 800x600 ✓
```

#### Scenario 4: Similar Aspect Ratio
```
Target: 800x600 (4:3 ratio)
Method: 1024x768 (4:3 ratio)

1. Scale to match: 800x600
2. No padding/cropping needed
Result: 800x600 ✓
```

## Options Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `target` | Path to target image (defines output dimensions) |
| `method` | Path to method image (to be resized) |

### Optional Arguments

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `-o, --output` | filename | `<method>_resized.ext` | Output filename |
| `--pad-color` | `R,G,B` or `none` | `0,0,0` (black) | Padding color |
| `--crop-mode` | `center`, `top`, `bottom`, `left`, `right` | `center` | Crop alignment |

### Padding Color Examples

```bash
# Black (default)
--pad-color 0,0,0

# White
--pad-color 255,255,255

# Gray
--pad-color 128,128,128

# Red
--pad-color 255,0,0

# Transparent (for RGBA images)
--pad-color none
```

### Crop Modes

When the scaled image is larger than the target:

- **`center`** - Crop equally from all sides
- **`top`** - Keep top portion, crop from bottom
- **`bottom`** - Keep bottom portion, crop from top
- **`left`** - Keep left portion, crop from right
- **`right`** - Keep right portion, crop from left

## Output

The program provides detailed information during processing:

```
Target image: 1920 x 1080
Method image: 1600 x 1200

Target aspect ratio: 1.778
Method aspect ratio: 1.333

Scaling mode: height-matched
Scale factor: 0.900
Scaled dimensions: 1440 x 1080

Padding needed: width=480, height=0
Padded with color: (0, 0, 0)
Padding: left=240, top=0, right=240, bottom=0

✓ Saved resized image to: method_resized.png
✓ Final dimensions: 1920 x 1080
✓ Matches target: True
```

## Supported Formats

**Input formats:**
- PNG, JPEG, JPG
- TIFF, TIF
- BMP, GIF, WEBP
- And all other formats supported by Pillow

**Output format:**
- Determined by output filename extension
- Preserves input format by default

## Quality Considerations

### Resampling Method

The program uses **LANCZOS resampling**, which provides:
- High quality for downscaling
- Good quality for upscaling
- Smooth edges and minimal artifacts

### Aspect Ratio

Aspect ratio is **always preserved** during the initial scaling step. This prevents:
- Distortion
- Stretching
- Squashing

Any dimensional mismatch is handled by padding or cropping, never by distorting the image.

## Use Cases

### 1. Dataset Preparation
```bash
# Resize all images to match a reference
for img in *.jpg; do
  python image_resizer.py reference.jpg "$img"
done
```

### 2. Thumbnail Generation
```bash
# Create thumbnails matching a template size
python image_resizer.py thumbnail_template.png large_photo.jpg -o thumbnail.jpg
```

### 3. Social Media Formatting
```bash
# Resize for specific platform dimensions
python image_resizer.py instagram_template_1080x1080.png my_photo.jpg
```

### 4. Document Processing
```bash
# Standardize document page sizes
python image_resizer.py template_page.tiff scanned_doc.tiff --pad-color 255,255,255
```

### 5. Batch Processing
```bash
#!/bin/bash
# Resize all images in a folder to match target
target="target_800x600.png"
for img in input/*.jpg; do
  name=$(basename "$img" .jpg)
  python image_resizer.py "$target" "$img" -o "output/${name}_resized.jpg"
done
```

## Troubleshooting

### "Image file could not be identified"
- Ensure the file is a valid image format
- Check file extension matches actual format

### "Output image has wrong dimensions"
- This should never happen - if it does, please report the bug
- The program verifies output dimensions automatically

### "Out of memory"
- Very large images may require significant RAM
- Try reducing image size first, or use a machine with more memory

### "Colors look wrong after resize"
- Ensure color space is appropriate (RGB, RGBA)
- Check if transparency is being handled correctly

## Tips & Best Practices

1. **Preserve quality**: Always resize from high-resolution to low-resolution when possible
2. **Use PNG for transparency**: If you need transparent padding, use PNG format
3. **Test with samples**: Try different crop modes and padding colors to find what works best
4. **Batch processing**: Write shell scripts for processing multiple images
5. **Check dimensions**: Use `identify` (ImageMagick) or `file` command to verify output

## Technical Details

- **Scaling algorithm**: Lanczos3 (high quality)
- **Padding alignment**: Centered by default
- **Crop alignment**: Configurable (center, edges)
- **Color spaces**: Preserves input color space (RGB, RGBA, etc.)
- **Precision**: Uses floating-point calculations for accurate scaling

## Requirements File

Create `requirements.txt`:
```
Pillow>=10.0.0
numpy>=1.24.0
```

## License

This utility script is provided as-is for image processing tasks.

## Version History

- **v1.0** - Initial release
  - Basic resizing with padding/cropping
  - Aspect ratio preservation
  - Configurable options
