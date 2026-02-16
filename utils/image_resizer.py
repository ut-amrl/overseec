#!/usr/bin/env python3
"""
Image Resizer - Match Target Dimensions
Resizes a method image to exactly match a target image's dimensions.
Handles aspect ratio differences with padding or cropping.
"""

import numpy as np
from PIL import Image
import sys
import argparse
from pathlib import Path


class ImageResizer:
    def __init__(self, target_path, method_path, pad_color=None, crop_mode='center'):
        """
        Initialize the image resizer.
        
        Args:
            target_path: Path to target image (defines output dimensions)
            method_path: Path to method image (to be resized)
            pad_color: Color for padding (R,G,B) tuple or None for transparent/black
            crop_mode: How to crop if needed - 'center', 'top', 'bottom', 'left', 'right'
        """
        self.target_path = Path(target_path)
        self.method_path = Path(method_path)
        self.pad_color = pad_color
        self.crop_mode = crop_mode
        
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_path}")
        if not self.method_path.exists():
            raise FileNotFoundError(f"Method image not found: {method_path}")
        
        # Load images
        self.target_img = Image.open(self.target_path)
        self.method_img = Image.open(self.method_path)
        
        # Get dimensions
        self.target_width, self.target_height = self.target_img.size
        self.method_width, self.method_height = self.method_img.size
        
        print(f"Target image: {self.target_width} x {self.target_height}")
        print(f"Method image: {self.method_width} x {self.method_height}")
        
    def resize_with_aspect_ratio(self):
        """
        Resize method image to match target dimensions.
        Preserves aspect ratio, then pads or crops to exact target size.
        """
        target_w, target_h = self.target_width, self.target_height
        method_w, method_h = self.method_width, self.method_height
        
        # Calculate aspect ratios
        target_ratio = target_w / target_h
        method_ratio = method_w / method_h
        
        print(f"\nTarget aspect ratio: {target_ratio:.3f}")
        print(f"Method aspect ratio: {method_ratio:.3f}")
        
        # Determine scaling strategy
        # Scale to fit within target dimensions while preserving aspect ratio
        if method_ratio > target_ratio:
            # Method is wider - scale to match width, may need padding/cropping on height
            scale_factor = target_w / method_w
            new_w = target_w
            new_h = int(method_h * scale_factor)
            scale_mode = "width-matched"
        else:
            # Method is taller - scale to match height, may need padding/cropping on width
            scale_factor = target_h / method_h
            new_w = int(method_w * scale_factor)
            new_h = target_h
            scale_mode = "height-matched"
        
        print(f"Scaling mode: {scale_mode}")
        print(f"Scale factor: {scale_factor:.3f}")
        print(f"Scaled dimensions: {new_w} x {new_h}")
        
        # Resize the method image
        resized = self.method_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Now handle padding or cropping to reach exact target dimensions
        if new_w == target_w and new_h == target_h:
            # Perfect match!
            print("Perfect match - no padding or cropping needed")
            return resized
        
        elif new_w < target_w or new_h < target_h:
            # Need padding
            print(f"Padding needed: width={target_w - new_w}, height={target_h - new_h}")
            return self._pad_image(resized, target_w, target_h)
        
        else:
            # Need cropping
            print(f"Cropping needed: width={new_w - target_w}, height={new_h - target_h}")
            return self._crop_image(resized, target_w, target_h)
    
    def _pad_image(self, img, target_w, target_h):
        """Add padding to reach target dimensions."""
        current_w, current_h = img.size
        
        # Determine padding color
        if self.pad_color is not None:
            # Use specified RGB color
            if img.mode == 'RGBA':
                pad_color = tuple(self.pad_color) + (255,)  # Add alpha
            else:
                pad_color = tuple(self.pad_color)
        else:
            # Use transparent for RGBA, black for RGB
            if img.mode == 'RGBA':
                pad_color = (0, 0, 0, 0)  # Transparent
            else:
                pad_color = (0, 0, 0)  # Black
        
        # Create new image with target dimensions
        result = Image.new(img.mode, (target_w, target_h), pad_color)
        
        # Calculate padding amounts (center the image)
        pad_left = (target_w - current_w) // 2
        pad_top = (target_h - current_h) // 2
        
        # Paste the resized image onto the padded canvas
        result.paste(img, (pad_left, pad_top))
        
        print(f"Padded with color: {pad_color}")
        print(f"Padding: left={pad_left}, top={pad_top}, right={target_w-current_w-pad_left}, bottom={target_h-current_h-pad_top}")
        
        return result
    
    def _crop_image(self, img, target_w, target_h):
        """Crop image to reach target dimensions."""
        current_w, current_h = img.size
        
        # Calculate crop box based on crop mode
        if self.crop_mode == 'center':
            left = (current_w - target_w) // 2
            top = (current_h - target_h) // 2
        elif self.crop_mode == 'top':
            left = (current_w - target_w) // 2
            top = 0
        elif self.crop_mode == 'bottom':
            left = (current_w - target_w) // 2
            top = current_h - target_h
        elif self.crop_mode == 'left':
            left = 0
            top = (current_h - target_h) // 2
        elif self.crop_mode == 'right':
            left = current_w - target_w
            top = (current_h - target_h) // 2
        else:
            # Default to center
            left = (current_w - target_w) // 2
            top = (current_h - target_h) // 2
        
        right = left + target_w
        bottom = top + target_h
        
        # Ensure we don't go out of bounds
        left = max(0, min(left, current_w - target_w))
        top = max(0, min(top, current_h - target_h))
        right = left + target_w
        bottom = top + target_h
        
        result = img.crop((left, top, right, bottom))
        
        print(f"Crop mode: {self.crop_mode}")
        print(f"Crop box: left={left}, top={top}, right={right}, bottom={bottom}")
        
        return result
    
    def save_resized(self, output_path=None):
        """Resize and save the method image."""
        if output_path is None:
            # Generate default output filename
            output_path = self.method_path.parent / f"{self.method_path.stem}_resized{self.method_path.suffix}"
        else:
            output_path = Path(output_path)
        
        print("\n" + "="*70)
        print("RESIZING IMAGE")
        print("="*70)
        
        # Perform the resize
        resized_img = self.resize_with_aspect_ratio()
        
        # Verify dimensions
        final_w, final_h = resized_img.size
        assert final_w == self.target_width and final_h == self.target_height, \
            f"Final dimensions {final_w}x{final_h} don't match target {self.target_width}x{self.target_height}"
        
        # Save the result
        resized_img.save(output_path)
        
        print("\n" + "="*70)
        print("RESIZE COMPLETE")
        print("="*70)
        print(f"✓ Saved resized image to: {output_path}")
        print(f"✓ Final dimensions: {final_w} x {final_h}")
        print(f"✓ Matches target: {final_w == self.target_width and final_h == self.target_height}")
        
        return output_path


def parse_color(color_str):
    """Parse color string in format 'R,G,B' to tuple."""
    if color_str is None or color_str.lower() == 'none':
        return None
    try:
        parts = color_str.split(',')
        if len(parts) != 3:
            raise ValueError()
        r, g, b = [int(x.strip()) for x in parts]
        if not all(0 <= c <= 255 for c in [r, g, b]):
            raise ValueError()
        return (r, g, b)
    except:
        raise argparse.ArgumentTypeError(
            "Color must be in format 'R,G,B' with values 0-255, or 'none'"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Resize method image to match target image dimensions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic resize (center crop/pad with black)
  python image_resizer.py target.png method.png
  
  # Specify output filename
  python image_resizer.py target.png method.png -o output.png
  
  # Use white padding instead of black
  python image_resizer.py target.png method.png --pad-color 255,255,255
  
  # Use transparent padding (for RGBA images)
  python image_resizer.py target.png method.png --pad-color none
  
  # Crop from top instead of center
  python image_resizer.py target.png method.png --crop-mode top
  
Crop modes: center, top, bottom, left, right
        """
    )
    
    parser.add_argument('target', help='Target image (defines output dimensions)')
    parser.add_argument('method', help='Method image (to be resized)')
    parser.add_argument('-o', '--output', help='Output filename (default: <method>_resized.ext)')
    parser.add_argument('--pad-color', type=parse_color, default=(0, 0, 0),
                       help='Padding color as R,G,B (e.g., "255,255,255" for white, "none" for transparent/black)')
    parser.add_argument('--crop-mode', choices=['center', 'top', 'bottom', 'left', 'right'],
                       default='center', help='How to crop if needed (default: center)')
    
    args = parser.parse_args()
    
    try:
        resizer = ImageResizer(
            target_path=args.target,
            method_path=args.method,
            pad_color=args.pad_color,
            crop_mode=args.crop_mode
        )
        
        output_path = resizer.save_resized(args.output)
        
        print("\nSuccess! Original files unchanged.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
