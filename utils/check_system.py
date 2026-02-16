#!/usr/bin/env python3
"""
Diagnostic script to check if the system can run the TIFF editor.
Checks for GUI backends and display availability.
"""

import sys
import os

print("="*70)
print("TIFF Editor - System Diagnostics")
print("="*70)
print()

# Check Python version
print("1. Python Version:")
print(f"   {sys.version}")
print()

# Check DISPLAY variable (for Linux/macOS)
print("2. Display Environment:")
display = os.environ.get('DISPLAY', 'NOT SET')
print(f"   DISPLAY = {display}")
if display == 'NOT SET' and sys.platform.startswith('linux'):
    print("   ⚠️  WARNING: DISPLAY not set (needed for Linux GUI)")
print()

# Check tkinter
print("3. Tkinter (Standard GUI Library):")
try:
    import tkinter
    print("   ✓ Tkinter is available")
    try:
        # Try to create a window (don't show it)
        root = tkinter.Tk()
        root.withdraw()
        root.destroy()
        print("   ✓ Can create Tkinter windows")
    except Exception as e:
        print(f"   ✗ Cannot create Tkinter windows: {e}")
except ImportError as e:
    print(f"   ✗ Tkinter NOT available: {e}")
    print("   → Install with: sudo apt-get install python3-tk (Ubuntu/Debian)")
print()

# Check matplotlib
print("4. Matplotlib:")
try:
    import matplotlib
    print(f"   ✓ Matplotlib version: {matplotlib.__version__}")
    print(f"   ✓ Current backend: {matplotlib.get_backend()}")
    
    # Check available backends
    available_backends = []
    test_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg', 'MacOSX']
    
    for backend in test_backends:
        try:
            matplotlib.use(backend, force=True)
            available_backends.append(backend)
        except:
            pass
    
    if available_backends:
        print(f"   ✓ Available interactive backends: {', '.join(available_backends)}")
    else:
        print("   ✗ No interactive backends available!")
    
    # Try to import pyplot
    try:
        import matplotlib.pyplot as plt
        print("   ✓ Can import matplotlib.pyplot")
    except Exception as e:
        print(f"   ✗ Cannot import pyplot: {e}")
        
except ImportError:
    print("   ✗ Matplotlib NOT installed")
    print("   → Install with: pip install matplotlib")
print()

# Check Pillow
print("5. Pillow (Image Library):")
try:
    from PIL import Image
    import PIL
    print(f"   ✓ Pillow version: {PIL.__version__}")
except ImportError:
    print("   ✗ Pillow NOT installed")
    print("   → Install with: pip install Pillow")
print()

# Check numpy
print("6. NumPy (Array Library):")
try:
    import numpy as np
    print(f"   ✓ NumPy version: {np.__version__}")
except ImportError:
    print("   ✗ NumPy NOT installed")
    print("   → Install with: pip install numpy")
print()

# Platform-specific checks
print("7. Platform-Specific Info:")
print(f"   Platform: {sys.platform}")
if sys.platform.startswith('linux'):
    print("   System: Linux")
    # Check for X11
    if os.path.exists('/tmp/.X11-unix'):
        print("   ✓ X11 socket exists")
    else:
        print("   ⚠️  X11 socket not found")
    
    # Check if in WSL
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                print("   ℹ️  Running in WSL - you need an X server on Windows")
                print("      (VcXsrv, Xming, or WSLg)")
    except:
        pass
        
elif sys.platform == 'darwin':
    print("   System: macOS")
    print("   ℹ️  Should work out of the box")
    
elif sys.platform.startswith('win'):
    print("   System: Windows")
    print("   ℹ️  Should work with standard Python installation")

print()

# Final verdict
print("="*70)
print("VERDICT:")
print("="*70)

errors = []
warnings = []

# Check critical components
try:
    import tkinter
except:
    errors.append("Tkinter not available - GUI won't work")

try:
    import matplotlib.pyplot
    backend = matplotlib.get_backend()
    if backend == 'agg' or not matplotlib.rcsetup.interactive_bk:
        errors.append("No interactive matplotlib backend available")
except:
    errors.append("Matplotlib not properly configured")

try:
    from PIL import Image
except:
    errors.append("Pillow not installed")

try:
    import numpy
except:
    errors.append("NumPy not installed")

if display == 'NOT SET' and sys.platform.startswith('linux'):
    warnings.append("DISPLAY not set - may cause issues on Linux")

if errors:
    print("❌ CANNOT RUN - Fix these issues:")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print()
    print("See TROUBLESHOOTING.md for solutions")
elif warnings:
    print("⚠️  MAY HAVE ISSUES:")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")
    print()
    print("Try running anyway, or see TROUBLESHOOTING.md")
else:
    print("✅ ALL CHECKS PASSED - Should work fine!")
    print()
    print("You can now run:")
    print("   python tiff_editor.py your_image.tiff")

print("="*70)
