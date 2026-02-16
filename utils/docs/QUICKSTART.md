# TIFF Editor - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 0: Quick System Check (30 seconds)

```bash
python3 check_system.py
```

If you see ❌ errors, you need to install `python3-tk`:

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Then re-run the check
python3 check_system.py
```

### Step 1: Setup (One-time)

**Option A - Automated Setup (Recommended)**

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```batch
setup.bat
```

**Option B - Manual Setup**

```bash
# Create virtual environment
python3 -m venv venv  # or 'python -m venv venv' on Windows

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Program

```bash
# Make sure virtual environment is activated!
python tiff_editor.py your_image.tiff
```

### Step 3: Interactive Workflow

The program will open windows for each step:

1. **Window 1**: Click to mark START (red marker) → Close window
2. **Window 2**: Click to mark GOAL (blue marker) → Close window
3. **Window 3**: Click & drag to draw PATH (green line) → Close window

Done! Your files are saved automatically.

---

## 📁 What Gets Created?

Starting with `my_image.tiff`, you get:

- **`my_image_edit_1.tiff`** - Image with red & blue markers
- **`my_image_user_path.tiff`** - Image with green path ONLY (no markers)
- **`my_image.tiff`** - ✓ Original unchanged!

**Note**: Output files match exactly what you see on screen!

---

## 🎨 Colors Used

- **Start**: Pure Red `RGB(255, 0, 0)`
- **Goal**: Pure Blue `RGB(0, 0, 255)`
- **Path**: Pure Green `RGB(0, 255, 0)`

---

## 🧪 Test It Out

Want to try it first?

```bash
# Create a test image
python create_test_image.py

# Run the editor on it
python tiff_editor.py test_image.tiff
```

---

## 💡 Pro Tips

- **Reposition markers**: Click multiple times before closing the window
- **Smooth paths**: Move mouse slowly while dragging
- **Skip a step**: Just close the window without clicking
- **Large images**: Windows will match image size

---

## ❓ Troubleshooting

**"FigureCanvasAgg is non-interactive"**
→ Install tkinter: `sudo apt-get install python3-tk`
→ Or see `TROUBLESHOOTING.md` for other solutions

**"ModuleNotFoundError"**
→ Did you activate the virtual environment?

**"File not found"**
→ Check the file path. Use quotes for names with spaces.

**Windows won't appear**
→ Check that you have a display/GUI environment.

---

## 📋 File Descriptions

| File | Purpose |
|------|---------|
| `tiff_editor.py` | Main program |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `setup.sh` | Linux/Mac setup script |
| `setup.bat` | Windows setup script |
| `create_test_image.py` | Generate test images |
| `QUICKSTART.md` | This guide |

---

## 🔄 Complete Workflow Example

```bash
# 1. Setup (first time only)
./setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Create test image
python create_test_image.py sample.tiff

# 4. Edit the image
python tiff_editor.py sample.tiff

# 5. Follow the interactive prompts!

# 6. Check your outputs
ls -l sample_*.tiff

# 7. Deactivate when done
deactivate
```

---

## Need More Help?

See `README.md` for detailed documentation including:
- Technical specifications
- Marker design details
- Advanced usage
- Troubleshooting guide
