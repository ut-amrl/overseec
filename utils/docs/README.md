# Interactive TIFF Editor

A Python program for interactively marking start/goal locations and drawing paths on multi-layer TIFF images.

## Features

1. **Load TIFF Files**: Reads RGB image from first layer
2. **Display True Dimensions**: Shows image at actual pixel dimensions
3. **Mark Start Location**: Interactive point selection with pure red (255,0,0) ring marker
4. **Mark Goal Location**: Interactive point selection with pure blue (0,0,255) ring marker
5. **Save Markers**: Outputs `<filename>_edit_1.tiff` with markers overlaid
6. **Draw Path**: Click and drag to create pure green (0,255,0) pixel path
7. **Save Path**: Outputs `<filename>_user_path.tiff` with complete visualization
8. **Safe Operation**: Original input file remains completely unchanged

## Installation

### Step 0: Check Your System (Recommended)

Before installing, run the diagnostic script to check if your system can run the GUI:

```bash
python3 check_system.py
```

This will tell you if you need to install any system packages (like `python3-tk`).

### Step 1: Create Virtual Environment

#### On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:
- `numpy` - Array operations and image manipulation
- `matplotlib` - Interactive visualization and user interface
- `Pillow` - TIFF file reading and writing

## Usage

### Configurable Parameters

You can customize the appearance of markers and paths by editing these values in `tiff_editor.py` (in the `__init__` method of the `TiffEditor` class):

```python
# Marker properties (configurable)
self.ring_radius = 10       # Radius of the ring in pixels
self.dot_radius = 3         # Radius of the center dot in pixels
self.ring_thickness = 2     # Thickness of the ring line in pixels
self.path_line_width = 2    # Width of the path line in pixels
```

**Note:** The output TIFF files will exactly match what you see in the display windows.

### Basic Usage

```bash
python tiff_editor.py <input_tiff_file>
```

Example:
```bash
python tiff_editor.py my_image.tiff
```

### Workflow

The program will guide you through these steps:

#### 1. Mark Start Location (Red)
- A window will open showing your image
- Click anywhere on the image to mark the start location
- A red ring with a center dot will appear
- You can click again to change the location
- Close the window when satisfied

#### 2. Mark Goal Location (Blue)
- A new window opens (showing the start marker if you placed one)
- Click to mark the goal location
- A blue ring with a center dot will appear
- You can click again to change the location
- Close the window when satisfied

#### 3. Auto-Save Markers
- The program automatically saves `<filename>_edit_1.tiff`
- This file contains the original image with red and blue markers

#### 4. Draw Path (Green)
- A new window opens showing both markers
- Click and hold the mouse button, then drag to draw
- Release to stop drawing
- Click and drag again to continue adding to the path
- The path is drawn in pure green (0,255,0)
- Close the window when finished

#### 5. Auto-Save Path
- The program automatically saves `<filename>_user_path.tiff`
- This file contains everything: original image, markers, and path

### Output Files

Given an input file `image.tiff`, the program creates:

1. **`image_edit_1.tiff`**
   - Original RGB image
   - Red ring marker at start location (if marked)
   - Blue ring marker at goal location (if marked)

2. **`image_user_path.tiff`**
   - Original RGB image
   - Green path drawn by user (ONLY - no markers)

**Important**: The original `image.tiff` is never modified!

**Note**: The markers and path in the output files will match exactly what you see in the display windows (same thickness, size, etc.).

## Color Specifications

- **Start Marker**: Pure Red RGB(255, 0, 0)
- **Goal Marker**: Pure Blue RGB(0, 0, 255)
- **Path**: Pure Green RGB(0, 255, 0)

## Marker Design

Each marker consists of:
- A **center dot** (default 3-pixel radius filled circle)
- An **outer ring** (default 10-pixel radius hollow circle)
- **Ring thickness** (default 2 pixels wide)

The path is drawn with:
- **Line width** (default 2 pixels wide)

These values can be customized in the code (see Configuration section above).

**Important**: The output TIFF images will show markers and paths with the exact same visual appearance as in the display windows - what you see is what you get!

## Tips

1. **Changing Markers**: Before closing a window, you can click multiple times to reposition a marker. Only the last click is saved.

2. **Path Drawing**: For smoother paths, move the mouse slowly while dragging. Fast movements may create gaps.

3. **Skipping Steps**: You can close windows without clicking to skip marking start/goal or drawing a path.

4. **Image Size**: The program displays images at their true pixel dimensions. For very large images, the window may be large.

## Troubleshooting

## Troubleshooting

### "FigureCanvasAgg is non-interactive" Error

This means matplotlib cannot find a GUI backend. **Most common solution:**

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**Other systems:** See `TROUBLESHOOTING.md` for detailed solutions.

### Run System Check

```bash
python3 check_system.py
```

This diagnostic tool will identify exactly what's missing.

### "File not found" Error
- Ensure the TIFF file path is correct
- Use quotes around filenames with spaces: `python tiff_editor.py "my image.tiff"`

### "No module named..." Error
- Make sure you activated the virtual environment
- Re-run `pip install -r requirements.txt`

### Window Not Appearing
- Some systems may need an X server or display configured
- On remote systems, ensure X11 forwarding is enabled

### Image Appears Distorted
- The program preserves the original image dimensions
- Make sure the first layer of your TIFF is an RGB image

## Deactivating Virtual Environment

When finished:
```bash
deactivate
```

## Technical Details

- **Image Format**: RGB (3-channel) TIFF files
- **Compression**: Output files use TIFF deflate compression
- **Coordinate System**: Standard image coordinates (0,0 at top-left)
- **Path Smoothing**: Uses Bresenham's line algorithm for continuous paths
- **Layer Management**: Markers and paths are managed separately in memory before compositing

## Requirements

- Python 3.7 or higher
- Operating system with display capability (GUI required)
- Sufficient memory for loading image into RAM

## License

This is a utility script provided as-is for image editing tasks.
