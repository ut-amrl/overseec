# Path Analysis Pipeline - Complete Guide for Ubuntu 24.04

A complete toolkit for drawing, resizing, and comparing paths in images. This pipeline consists of three integrated programs that work together to analyze and compare paths.

---

## 📦 What's in This Toolkit

### 1. TIFF Editor (`tiff_editor.py`)
Interactive tool for drawing paths on images with mouse input.
- Mark start and goal locations with colored markers
- Draw paths by clicking and dragging
- Outputs images with pure color paths for analysis

### 2. Image Resizer (`image_resizer.py`)
Resize images to match exact target dimensions while preserving aspect ratio.
- Smart padding and cropping
- High-quality LANCZOS resampling
- Saves detailed statistics to text file

### 3. Path Comparator (`path_comparator.py`)
Calculate the unified directed Hausdorff distance between two paths.
- Extracts pure color paths as binary masks
- Computes distance metric in pixels
- Quantifies how well paths match

---

## 🚀 Getting Started

### Step 1: System Check (Optional but Recommended)

Before installation, verify your system has the necessary GUI support:

```bash
python3 check_system.py
```

**What to look for:**
- ✅ All checks should pass (green checkmarks)
- ❌ If "Tkinter NOT available", install it first (see below)

### Step 2: Install System Dependencies

The TIFF Editor requires `tkinter` for the interactive GUI:

```bash
sudo apt-get update
sudo apt-get install python3-tk
```

**Verify installation:**
```bash
python3 -c "import tkinter; print('Tkinter OK')"
```

### Step 3: Run Automated Setup

Use the provided setup script to create a virtual environment and install all Python dependencies:

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup
./setup.sh
```

**What this does:**
1. Creates a Python virtual environment (`venv/`)
2. Upgrades pip to the latest version
3. Installs all required packages:
   - numpy >= 1.24.0
   - matplotlib >= 3.7.0
   - Pillow >= 10.0.0

**Expected output:**
```
=========================================
TIFF Editor - Setup Script
=========================================

Checking Python installation...
Found: python3
Python 3.12.x

Creating virtual environment...
Virtual environment created successfully!

Installing dependencies...
...
Setup Complete!
=========================================
```

### Step 4: Activate the Virtual Environment

Every time you want to use the tools, activate the virtual environment:

```bash
source venv/bin/activate
```

**You'll see** `(venv)` appear in your terminal prompt:
```
(venv) user@ubuntu:~$
```

---

## 🎯 Complete Workflow: Comparing Two Paths

### Overview

This workflow compares a **target path** (ground truth) against a **method path** (algorithm output or test result).

**Color Convention:**
- Target path: **Pure Green** RGB(0, 255, 0)
- Method path: **Pure Red** RGB(255, 0, 0)

---

### Part 1: Create Target Image with Green Path

**Input:** Your target/reference image (e.g., `target.tiff`)

**Goal:** Draw the reference/ground truth path in pure green

#### 1.1 Launch TIFF Editor

```bash
python tiff_editor.py target.tiff
```

#### 1.2 Interactive Steps

The program opens three interactive windows in sequence:

**Window 1 - Mark Start Location:**
- A window shows your image
- Click once to mark the start location
- A red ring with center dot appears
- You can click again to reposition
- Close the window when satisfied

**Window 2 - Mark Goal Location:**
- A new window opens (showing the start marker)
- Click once to mark the goal location
- A blue ring with center dot appears
- You can click again to reposition
- Close the window when satisfied

**Window 3 - Draw Path:**
- A new window opens (showing both markers)
- Click and hold the left mouse button
- Drag to draw the path in **pure green**
- Release to stop drawing
- Click and drag again to continue
- Close the window when finished

#### 1.3 Output Files

The program creates two files:
- `target_edit_1.tiff` - Image with start/goal markers only
- `target_user_path.tiff` - Image with markers AND green path ← **Use this one**

**Terminal output:**
```
Processing complete!
Original file unchanged.
```

---

### Part 2: Create Method Image with Red Path

**Important Note:** The TIFF Editor draws paths in green. For the method image, you need a **red path**. You have two options:

#### Option A: Use an Image Editor
1. Run the TIFF Editor on your method image to draw the path
2. Use GIMP or another image editor to change the green path to pure red RGB(255,0,0)

#### Option B: Programmatically Generate Red Path
If your method is an algorithm, have it output an image with the path drawn in pure red RGB(255,0,0).

**Result:** You should have `method_with_red_path.tiff` containing:
- Your method image
- A path drawn in pure red RGB(255, 0, 0)

---

### Part 3: Resize Method Image to Match Target

Images must have identical dimensions for comparison.

```bash
python image_resizer.py target_user_path.tiff method_with_red_path.tiff
```

#### What Happens:

**Terminal output:**
```
Target image: 800 x 600
Method image: 1024 x 768

Target aspect ratio: 1.333
Method aspect ratio: 1.333

Scaling mode: width-matched
Scale factor: 0.781
Scaled dimensions: 800 x 600

Perfect match - no padding or cropping needed

======================================================================
RESIZE COMPLETE
======================================================================
✓ Saved resized image to: method_with_red_path_resized.tiff
✓ Final dimensions: 800 x 600
✓ Matches target: True
✓ Saved resize statistics to: method_with_red_path_resized_resize_stats.txt
```

#### Output Files:
- `method_with_red_path_resized.tiff` - Resized method image (exact same dimensions as target)
- `method_with_red_path_resized_resize_stats.txt` - Detailed statistics about the resize operation

**Important:** Always check that "Matches target: True" appears in the output!

---

### Part 4: Compare Paths with Hausdorff Distance

Now compare the green path (target) against the red path (method):

```bash
python path_comparator.py target_user_path.tiff method_with_red_path_resized.tiff
```

#### What Happens:

**Terminal output:**
```
Target image loaded: (600, 800, 3)
Method image loaded: (600, 800, 3)

======================================================================
EXTRACTING PATHS
======================================================================

Target image (green path):
  Pure green pixels found: 1234

Method image (red path):
  Pure red pixels found: 1567

======================================================================
CALCULATING HAUSDORFF DISTANCE
======================================================================
Method points: 1567
Target points: 1234

======================================================================
RESULT
======================================================================

Unified Directed Hausdorff Distance: 12.3456 pixels

Interpretation:
  - The maximum minimum distance from any red path pixel
    to the nearest green path pixel is 12.3456 pixels.
  - Lower values indicate better path matching.
  - A value of 0 means paths overlap perfectly.
======================================================================
```

#### Understanding the Result

| Distance (pixels) | Match Quality |
|-------------------|---------------|
| 0.0 | Perfect - paths overlap completely |
| 0.1 - 5.0 | Excellent - very close match |
| 5.0 - 15.0 | Good - paths follow similar trajectory |
| 15.0 - 30.0 | Fair - noticeable deviation |
| > 30.0 | Poor - significant differences |

**The distance tells you:** The worst-case distance from any pixel in the red path to the nearest pixel in the green path.

---

### Part 5: Debug and Visualize (Optional)

If you want to see exactly what paths were extracted:

```bash
python path_comparator.py target_user_path.tiff method_with_red_path_resized.tiff --save-masks
```

**Additional output files:**
- `target_user_path_green_mask.png` - Binary mask showing green path (white pixels on black background)
- `method_with_red_path_resized_red_mask.png` - Binary mask showing red path (white pixels on black background)

**Use these to:**
- Verify paths were detected correctly
- Debug "No pixels found" errors
- Visualize what the algorithm is comparing

---

## 🧪 Testing with Sample Images

### Generate Test Images

We provide scripts to create test images for trying out the tools:

#### For TIFF Editor Testing:
```bash
python create_test_image.py test_image.tiff
```
Creates a colorful test image with gradients and grid lines.

#### For Image Resizer Testing:
```bash
python create_test_images_resizer.py
```
Creates multiple test images:
- `target_800x600.png` - Target reference size
- `method_wide.png` - Wide image (tests padding)
- `method_tall.png` - Tall image (tests padding)
- `method_square.png` - Square image
- `method_panorama.png` - Very wide panoramic image

**Try the resizer:**
```bash
python image_resizer.py target_800x600.png method_wide.png
python image_resizer.py target_800x600.png method_tall.png
```

---

## 📋 Quick Reference Commands

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Draw Path on Target Image
```bash
python tiff_editor.py target.tiff
# → Creates: target_user_path.tiff (with green path)
```

### Resize Method to Match Target
```bash
python image_resizer.py target_user_path.tiff method_red.tiff
# → Creates: method_red_resized.tiff
```

### Compare Paths
```bash
python path_comparator.py target_user_path.tiff method_red_resized.tiff
# → Outputs: Hausdorff distance
```

### Deactivate Virtual Environment
```bash
deactivate
```

---

## 🔧 Troubleshooting

### Problem: "FigureCanvasAgg is non-interactive"

**Cause:** Tkinter is not installed

**Fix:**
```bash
sudo apt-get install python3-tk
python3 check_system.py  # Verify installation
```

### Problem: "No pure green pixels found" or "No pure red pixels found"

**Causes:**
- Wrong color used (not exactly RGB 0,255,0 or 255,0,0)
- Image was saved as JPEG (compression changes colors)
- Using the wrong output file

**Fixes:**
1. **Verify you're using the right file:**
   - Target: Use `target_user_path.tiff` (not `target_edit_1.tiff`)
   - Method: Ensure red path is pure RGB(255, 0, 0)

2. **Check colors programmatically:**
```bash
python3 << 'EOF'
from PIL import Image
import numpy as np

img = Image.open('your_image.tiff')
arr = np.array(img.convert('RGB'))

green = np.sum(np.all(arr == [0, 255, 0], axis=2))
red = np.sum(np.all(arr == [255, 0, 0], axis=2))

print(f"Pure green pixels: {green}")
print(f"Pure red pixels: {red}")
EOF
```

3. **Always use PNG or TIFF:** Never use JPEG for colored paths

### Problem: "Image dimensions don't match"

**Cause:** Method image not resized to match target

**Fix:**
```bash
# Always run the resizer before comparing
python image_resizer.py target_user_path.tiff method.tiff
python path_comparator.py target_user_path.tiff method_resized.tiff
```

### Problem: GUI windows don't appear

**Cause:** DISPLAY not set or X11 not available

**Fix:**
```bash
# Check DISPLAY variable
echo $DISPLAY

# Should show something like :0 or :1
# If empty, you may be in a headless environment or SSH without X11 forwarding
```

---

## 💡 Best Practices

### 1. File Organization
```
project/
├── originals/
│   ├── target.tiff
│   └── method.tiff
├── processed/
│   ├── target_user_path.tiff       # Green path
│   ├── method_red_path.tiff        # Red path
│   └── method_red_path_resized.tiff
└── results/
    ├── method_red_path_resized_resize_stats.txt
    └── comparison_results.txt
```

### 2. Always Use Lossless Formats
- ✅ PNG - Lossless, widely supported
- ✅ TIFF - Lossless, good for scientific data
- ❌ JPEG - Lossy compression changes colors

### 3. Keep Original Files Safe
All programs create new files and never modify the originals.

### 4. Save Your Results
```bash
# Redirect output to file
python path_comparator.py target.tiff method.tiff > results.txt

# Or append to existing file
python path_comparator.py target.tiff method.tiff >> all_results.txt
```

### 5. Use Descriptive Filenames
```bash
# Good
algorithm_dijkstra_output.tiff
expert_ground_truth.tiff

# Poor
output.tiff
temp.tiff
```

---

## 📊 Example Complete Workflow

Here's a complete example from start to finish:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Draw ground truth path (green) on target
python tiff_editor.py maze_map.tiff
# → Interactive: mark start, goal, draw path
# → Creates: maze_map_user_path.tiff

# 3. You have a method image with red path from your algorithm
# Let's say it's: algorithm_output_red.tiff

# 4. Resize method to match target dimensions
python image_resizer.py maze_map_user_path.tiff algorithm_output_red.tiff
# → Creates: algorithm_output_red_resized.tiff

# 5. Compare paths
python path_comparator.py maze_map_user_path.tiff algorithm_output_red_resized.tiff > results.txt

# 6. Check results
cat results.txt
# Shows: Unified Directed Hausdorff Distance: X.XXXX pixels

# 7. Deactivate when done
deactivate
```

---

## 🎨 Color Requirements Reference

**Critical:** All programs use pure colors with exact RGB values!

| Component | Color | RGB Values | Notes |
|-----------|-------|------------|-------|
| Start Marker | Red | (255, 0, 0) | TIFF Editor output |
| Goal Marker | Blue | (0, 0, 255) | TIFF Editor output |
| Target Path | **Green** | **(0, 255, 0)** | **For path_comparator** |
| Method Path | **Red** | **(255, 0, 0)** | **For path_comparator** |

**Why exact colors?**
- Path extraction uses exact pixel matching
- Even RGB(0, 254, 0) won't be detected
- JPEG compression can change colors slightly
- Always use PNG or TIFF

---

## 📦 Dependencies

All dependencies are installed by `setup.sh`:

```
numpy >= 1.24.0      # Array operations
matplotlib >= 3.7.0  # Interactive GUI (TIFF Editor)
Pillow >= 10.0.0     # Image processing
```

**System package** (install separately):
```
python3-tk           # GUI support for matplotlib
```

---

## 📚 Additional Documentation

Detailed documentation for each program:
- `IMAGE_RESIZER_README.md` - Image Resizer detailed docs
- `COMPARATOR_README.md` - Path Comparator detailed docs
- `WORKFLOW_GUIDE.md` - Advanced workflows and use cases
- `TROUBLESHOOTING.md` - GUI backend troubleshooting

Quick start guides:
- `QUICKSTART.md` - TIFF Editor quick start
- `RESIZER_QUICKSTART.md` - Image Resizer quick start
- `COMPARATOR_QUICKSTART.md` - Path Comparator quick start

---

## 🆘 Need Help?

### Quick Diagnostics
```bash
# Check system compatibility
python3 check_system.py

# Check if virtual environment is active
which python
# Should show: /path/to/venv/bin/python

# Check installed packages
pip list | grep -E "numpy|matplotlib|Pillow"
```

### Common Issues Checklist

- [ ] Virtual environment activated? (`source venv/bin/activate`)
- [ ] Tkinter installed? (`sudo apt-get install python3-tk`)
- [ ] Using correct output files? (e.g., `*_user_path.tiff`)
- [ ] Pure colors? (Green: 0,255,0 | Red: 255,0,0)
- [ ] Same dimensions? (run resizer first)
- [ ] PNG or TIFF format? (not JPEG)

---

## 🎉 You're Ready!

Your complete path analysis pipeline is set up and ready to use. Start with:

```bash
source venv/bin/activate
python tiff_editor.py your_image.tiff
```

Happy path analyzing! 🛤️
