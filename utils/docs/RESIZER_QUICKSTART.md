# Image Resizer - Quick Start Guide

## 🚀 Get Started in 60 Seconds

### 1. Install (One-time)
```bash
pip install Pillow numpy
```

### 2. Basic Usage
```bash
python image_resizer.py target.png method.png
```
That's it! Creates `method_resized.png` matching `target.png` dimensions.

---

## 📖 What Does It Do?

Takes two images:
- **Target image**: Defines the output size you want
- **Method image**: The image you want to resize

Outputs:
- **`<method>_resized.ext`**: Method image resized to exactly match target dimensions

**Smart resizing:**
- ✅ Preserves aspect ratio (no distortion)
- ✅ Pads with color if too small
- ✅ Crops intelligently if too large

---

## 🎯 Common Commands

### Default (Black Padding, Center Crop)
```bash
python image_resizer.py target.png method.png
```

### Custom Output Name
```bash
python image_resizer.py target.png method.png -o my_output.png
```

### White Padding
```bash
python image_resizer.py target.png method.png --pad-color 255,255,255
```

### Transparent Padding (PNG)
```bash
python image_resizer.py target.png method.png --pad-color none
```

### Crop from Top
```bash
python image_resizer.py target.png method.png --crop-mode top
```

---

## 🧪 Test It

### Generate Test Images
```bash
python create_test_images_resizer.py
```

Creates:
- `target_800x600.png` - Your target size
- `method_wide.png` - Wider image (needs vertical padding)
- `method_tall.png` - Taller image (needs horizontal padding)
- `method_square.png` - Square image
- `method_panorama.png` - Very wide image

### Run Tests
```bash
# Test with wide image
python image_resizer.py target_800x600.png method_wide.png

# Test with white padding
python image_resizer.py target_800x600.png method_tall.png --pad-color 255,255,255

# Test all at once
for img in method_*.png; do
  python image_resizer.py target_800x600.png "$img"
done
```

---

## 🎨 Padding Colors

| Color | Command |
|-------|---------|
| Black (default) | `--pad-color 0,0,0` |
| White | `--pad-color 255,255,255` |
| Gray | `--pad-color 128,128,128` |
| Red | `--pad-color 255,0,0` |
| Transparent | `--pad-color none` |

---

## ✂️ Crop Modes

When image is too large after scaling:

| Mode | Effect |
|------|--------|
| `center` (default) | Crop equally from all sides |
| `top` | Keep top, remove bottom |
| `bottom` | Keep bottom, remove top |
| `left` | Keep left, remove right |
| `right` | Keep right, remove left |

---

## 💡 Real-World Examples

### Standardize Photo Sizes
```bash
# All photos to 1920x1080
for photo in *.jpg; do
  python image_resizer.py reference_1920x1080.jpg "$photo"
done
```

### Create Thumbnails
```bash
python image_resizer.py thumbnail_template.png large_photo.jpg -o thumb.jpg
```

### Prepare ML Dataset
```bash
# Resize all images to 224x224 (common ML input size)
for img in dataset/*.png; do
  name=$(basename "$img")
  python image_resizer.py template_224x224.png "$img" -o "processed/$name"
done
```

---

## 📋 How It Works

1. **Load images**: Read target and method images
2. **Calculate scaling**: Preserve aspect ratio while fitting to target
3. **Resize**: High-quality LANCZOS resampling
4. **Adjust**:
   - If too small → Pad with color
   - If too large → Crop from edges
5. **Save**: Output matches target dimensions exactly

### Visual Example

```
Target: 800x600          Method: 1600x900
    
    ┌─────────┐             ┌──────────────┐
    │         │             │              │
    │ 800x600 │    +        │  1600x900    │
    │         │             │              │
    └─────────┘             └──────────────┘
                                   ↓
                            Scale to 800x450
                                   ↓
                         Pad 75px top & bottom
                                   ↓
                            Final: 800x600 ✓
```

---

## ❓ FAQ

**Q: Will this distort my images?**  
A: No! Aspect ratio is always preserved. Any size difference is handled by padding or cropping.

**Q: What if images are already the same size?**  
A: They're simply copied (minimal processing).

**Q: Can I batch process?**  
A: Yes! Use a shell loop or write a batch script.

**Q: What formats are supported?**  
A: All common formats: PNG, JPEG, TIFF, BMP, GIF, WebP, etc.

**Q: What's the output quality?**  
A: Uses LANCZOS resampling (highest quality). Perfect for both upscaling and downscaling.

---

## 📚 Need More Info?

- **Full documentation**: `IMAGE_RESIZER_README.md`
- **Help command**: `python image_resizer.py --help`
- **Examples**: See test images output

---

## ⚡ Quick Reference Card

```bash
# Syntax
python image_resizer.py <target> <method> [options]

# Options
-o FILE              # Custom output filename
--pad-color R,G,B    # Padding color (e.g., 255,255,255)
--pad-color none     # Transparent padding
--crop-mode MODE     # center|top|bottom|left|right

# Examples
python image_resizer.py ref.png input.jpg                    # Basic
python image_resizer.py ref.png input.jpg -o out.png         # Custom name
python image_resizer.py ref.png input.jpg --pad-color 255,255,255  # White pad
python image_resizer.py ref.png input.jpg --crop-mode top    # Top crop
```

---

**That's it!** You're ready to resize images. 🎉
