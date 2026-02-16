#!/bin/bash
# Quick setup script for TIFF Editor

echo "========================================="
echo "TIFF Editor - Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python is not installed!"
    exit 1
fi

echo "Found: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully!"
echo ""

# Activate virtual environment and install dependencies
echo "Installing dependencies..."

# Detect OS for activation
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To use the TIFF editor:"
echo ""
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "1. Activate the environment:"
    echo "   venv\\Scripts\\activate"
else
    echo "1. Activate the environment:"
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. Run the editor:"
echo "   python tiff_editor.py your_image.tiff"
echo ""
echo "3. When done, deactivate:"
echo "   deactivate"
echo ""
echo "========================================="
