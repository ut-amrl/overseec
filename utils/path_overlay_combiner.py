#!/usr/bin/env python3
"""
Path Overlay Combiner
Combines multiple colored paths and markers onto a single background image.
Extracts and overlays:
- Red path from method image
- Green paths from one or more target images
- Start marker (red ring/dot) from target images
- Goal marker (blue ring/dot) from target images
All overlaid on a specified background image.
"""

import numpy as np
from PIL import Image
import sys
import argparse
from pathlib import Path


class PathOverlayCombiner:
    def __init__(self, background_path, target_paths, method_paths, 
                 target_inflation=0, method_inflation=0, yellow_start=False,
                 blue_target=False, green_goal=False):
        """
        Initialize the path overlay combiner.
        
        Args:
            background_path: Path to background image
            target_paths: List of paths to target images (with green paths and markers)
            method_paths: List of paths to method images (with red paths)
            target_inflation: Pixels to inflate target paths (default: 0)
            method_inflation: Pixels to inflate method paths (default: 0)
            yellow_start: If True, change start marker to yellow instead of red (default: False)
            blue_target: If True, change target path to blue instead of green (default: False)
            green_goal: If True, change goal marker to green instead of blue (default: False)
        """
        self.background_path = Path(background_path)
        
        # Convert to list if single path provided
        if isinstance(target_paths, (str, Path)):
            target_paths = [target_paths]
        if isinstance(method_paths, (str, Path)):
            method_paths = [method_paths]
        
        self.target_paths = [Path(p) for p in target_paths]
        self.method_paths = [Path(p) for p in method_paths]
        
        # Store inflation factors and color options
        self.target_inflation = target_inflation
        self.method_inflation = method_inflation
        self.yellow_start = yellow_start
        self.blue_target = blue_target
        self.green_goal = green_goal
        
        # Verify all files exist
        if not self.background_path.exists():
            raise FileNotFoundError(f"Background image not found: {background_path}")
        
        for target_path in self.target_paths:
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
        
        for method_path in self.method_paths:
            if not method_path.exists():
                raise FileNotFoundError(f"Method image not found: {method_path}")
        
        # Load background image
        self.background_img = Image.open(self.background_path)
        self.background_array = np.array(self.background_img.convert('RGB'))
        
        print(f"Background image loaded: {self.background_array.shape}")
        
        # Load target images
        self.target_arrays = []
        print(f"\nLoading {len(self.target_paths)} target image(s):")
        for i, target_path in enumerate(self.target_paths, 1):
            target_img = Image.open(target_path)
            target_array = np.array(target_img.convert('RGB'))
            self.target_arrays.append(target_array)
            print(f"  {i}. {target_path.name}: {target_array.shape}")
        
        # Load method images
        self.method_arrays = []
        print(f"\nLoading {len(self.method_paths)} method image(s):")
        for i, method_path in enumerate(self.method_paths, 1):
            method_img = Image.open(method_path)
            method_array = np.array(method_img.convert('RGB'))
            self.method_arrays.append(method_array)
            print(f"  {i}. {method_path.name}: {method_array.shape}")
        
        # Verify all dimensions match
        bg_shape = self.background_array.shape
        
        for i, target_array in enumerate(self.target_arrays, 1):
            if target_array.shape != bg_shape:
                print(f"\nWARNING: Target {i} dimensions don't match background!")
                print(f"  Background: {bg_shape}")
                print(f"  Target {i}: {target_array.shape}")
        
        for i, method_array in enumerate(self.method_arrays, 1):
            if method_array.shape != bg_shape:
                print(f"\nWARNING: Method {i} dimensions don't match background!")
                print(f"  Background: {bg_shape}")
                print(f"  Method {i}: {method_array.shape}")
    
    def extract_color_mask(self, image_array, color):
        """
        Extract binary mask for a specific pure color.
        
        Args:
            image_array: numpy array of image
            color: RGB tuple (e.g., (255, 0, 0) for red)
            
        Returns:
            Binary mask where color pixels = 1, others = 0
        """
        color_array = np.array(color)
        mask = np.all(image_array == color_array, axis=2).astype(np.uint8)
        return mask
    
    def inflate_mask(self, mask, inflation):
        """
        Inflate (dilate) a binary mask by a given number of pixels.
        
        Args:
            mask: Binary mask (2D numpy array)
            inflation: Number of pixels to inflate
            
        Returns:
            Inflated binary mask
        """
        if inflation <= 0:
            return mask
        
        # Get coordinates of all mask pixels
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            return mask
        
        # Create new mask
        inflated = mask.copy()
        
        # For each pixel in the mask, set surrounding pixels within inflation radius
        for y, x in coords:
            for dy in range(-inflation, inflation + 1):
                for dx in range(-inflation, inflation + 1):
                    # Use circular inflation (Euclidean distance)
                    if dy*dy + dx*dx <= inflation*inflation:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                            inflated[ny, nx] = 1
        
        return inflated
    
    def combine_overlays(self):
        """
        Combine all colored elements onto the background image.
        
        Returns:
            numpy.ndarray: Combined image with all overlays
        """
        # Start with a copy of the background
        result = self.background_array.copy()
        
        print("\n" + "="*70)
        print("EXTRACTING AND COMBINING COLORED ELEMENTS")
        print("="*70)
        
        if self.target_inflation > 0:
            print(f"Target path inflation: {self.target_inflation} pixels")
        if self.method_inflation > 0:
            print(f"Method path inflation: {self.method_inflation} pixels")
        if self.yellow_start:
            print(f"Start marker color: yellow (255,255,0)")
        if self.blue_target:
            print(f"Target path color: blue (0,0,255)")
        if self.green_goal:
            print(f"Goal marker color: green (0,255,0)")
        
        # Define colors
        pure_red = (255, 0, 0)      # Start marker and method path
        pure_blue = (0, 0, 255)     # Goal marker (or target path if swapped)
        pure_green = (0, 255, 0)    # Target paths (or goal marker if swapped)
        pure_yellow = (255, 255, 0) # Optional start marker color
        
        # Determine actual colors based on flags
        target_path_color = pure_blue if self.blue_target else pure_green
        goal_marker_color = pure_green if self.green_goal else pure_blue
        
        # Extract and overlay green paths from all targets
        total_green_pixels = 0
        total_green_pixels_inflated = 0
        
        target_color_name = "blue" if self.blue_target else "green"
        
        for i, target_array in enumerate(self.target_arrays, 1):
            green_mask = self.extract_color_mask(target_array, pure_green)
            green_count = np.sum(green_mask)
            total_green_pixels += green_count
            
            # Inflate if requested
            if self.target_inflation > 0:
                green_mask = self.inflate_mask(green_mask, self.target_inflation)
                green_count_inflated = np.sum(green_mask)
                total_green_pixels_inflated += green_count_inflated
            
            # Overlay target path with selected color
            result[green_mask > 0] = target_path_color
            
            print(f"\nTarget {i}:")
            print(f"  Target path pixels: {green_count}")
            if self.target_inflation > 0:
                print(f"  Target path pixels (after inflation): {green_count_inflated}")
        
        if self.target_inflation > 0:
            print(f"\nTotal target path pixels (original): {total_green_pixels}")
            print(f"Total target path pixels (inflated, {target_color_name}): {total_green_pixels_inflated}")
        else:
            print(f"\nTotal target path pixels ({target_color_name}): {total_green_pixels}")
        
        # Extract and overlay red paths from all methods
        total_red_path_pixels = 0
        total_red_path_pixels_inflated = 0
        
        for i, method_array in enumerate(self.method_arrays, 1):
            red_path_mask = self.extract_color_mask(method_array, pure_red)
            red_path_count = np.sum(red_path_mask)
            total_red_path_pixels += red_path_count
            
            # Inflate if requested
            if self.method_inflation > 0:
                red_path_mask = self.inflate_mask(red_path_mask, self.method_inflation)
                red_path_count_inflated = np.sum(red_path_mask)
                total_red_path_pixels_inflated += red_path_count_inflated
            
            # Overlay red path
            result[red_path_mask > 0] = pure_red
            
            print(f"\nMethod {i}:")
            print(f"  Red path pixels: {red_path_count}")
            if self.method_inflation > 0:
                print(f"  Red path pixels (after inflation): {red_path_count_inflated}")
        
        if self.method_inflation > 0:
            print(f"\nTotal red path pixels (original): {total_red_path_pixels}")
            print(f"Total red path pixels (inflated): {total_red_path_pixels_inflated}")
        else:
            print(f"\nTotal red path pixels: {total_red_path_pixels}")
        
        # Extract and overlay markers from all targets
        total_red_marker_pixels = 0
        total_blue_marker_pixels = 0
        
        start_marker_color_name = "yellow" if self.yellow_start else "red"
        goal_marker_color_name = "green" if self.green_goal else "blue"
        
        for i, target_array in enumerate(self.target_arrays, 1):
            red_mask = self.extract_color_mask(target_array, pure_red)
            blue_mask = self.extract_color_mask(target_array, pure_blue)
            
            red_marker_count = np.sum(red_mask)
            blue_marker_count = np.sum(blue_mask)
            
            total_red_marker_pixels += red_marker_count
            total_blue_marker_pixels += blue_marker_count
            
            # Overlay start marker
            if self.yellow_start:
                result[red_mask > 0] = pure_yellow
            else:
                result[red_mask > 0] = pure_red
            
            # Overlay goal marker
            result[blue_mask > 0] = goal_marker_color
            
            if red_marker_count > 0 or blue_marker_count > 0:
                print(f"\nTarget {i} markers:")
                if red_marker_count > 0:
                    print(f"  Start marker ({start_marker_color_name}) pixels: {red_marker_count}")
                if blue_marker_count > 0:
                    print(f"  Goal marker ({goal_marker_color_name}) pixels: {blue_marker_count}")
        
        print(f"\n" + "="*70)
        print("OVERLAY SUMMARY")
        print("="*70)
        print(f"Background image: {self.background_path.name}")
        
        if self.target_inflation > 0:
            print(f"Target path pixels ({target_color_name}, from {len(self.target_paths)} target(s), inflated by {self.target_inflation}): {total_green_pixels_inflated}")
        else:
            print(f"Target path pixels ({target_color_name}, from {len(self.target_paths)} target(s)): {total_green_pixels}")
        
        if self.method_inflation > 0:
            print(f"Method path pixels (red, from {len(self.method_paths)} method(s), inflated by {self.method_inflation}): {total_red_path_pixels_inflated}")
        else:
            print(f"Method path pixels (red, from {len(self.method_paths)} method(s)): {total_red_path_pixels}")
        
        print(f"Start marker pixels ({start_marker_color_name}): {total_red_marker_pixels}")
        print(f"Goal marker pixels ({goal_marker_color_name}): {total_blue_marker_pixels}")
        
        return result
    
    def save_combined(self, output_path=None):
        """
        Create and save the combined overlay image.
        
        Args:
            output_path: Path for output file (default: background_name_combined.ext)
            
        Returns:
            Path: Path to saved file
        """
        if output_path is None:
            output_path = self.background_path.parent / f"{self.background_path.stem}_combined{self.background_path.suffix}"
        else:
            output_path = Path(output_path)
        
        # Combine all overlays
        combined_array = self.combine_overlays()
        
        # Convert to PIL Image and save
        combined_img = Image.fromarray(combined_array, mode='RGB')
        combined_img.save(output_path)
        
        print(f"\n✓ Saved combined image to: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Combine colored paths and markers onto a background image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target, single method (default)
  python path_overlay_combiner.py background.tiff target.tiff method.tiff
  
  # Multiple targets, single method
  python path_overlay_combiner.py background.tiff target1.tiff target2.tiff method.tiff
  
  # Single target, multiple methods
  python path_overlay_combiner.py background.tiff target.tiff method1.tiff method2.tiff --num-methods 2
  
  # Multiple targets, multiple methods
  python path_overlay_combiner.py background.tiff t1.tiff t2.tiff m1.tiff m2.tiff m3.tiff --num-methods 3
  
  # With path inflation for visibility
  python path_overlay_combiner.py background.tiff target.tiff method.tiff \
      --target-inflation 2 --method-inflation 3
  
  # Color customization options
  python path_overlay_combiner.py background.tiff target.tiff method.tiff \
      --yellow-start --blue-target --green-goal
  
  # Everything together
  python path_overlay_combiner.py background.tiff t1.tiff t2.tiff m1.tiff m2.tiff \
      --num-methods 2 --target-inflation 2 --method-inflation 2 \
      --yellow-start --blue-target -o result.tiff

How --num-methods works:
  The last N images in the list are treated as method images.
  All remaining images (after background) are treated as targets.
  
  Example: background.tiff A.tiff B.tiff C.tiff D.tiff --num-methods 2
    → Targets: A.tiff, B.tiff
    → Methods: C.tiff, D.tiff

What this does:
  1. Starts with the background image (your original image)
  2. Overlays target paths from all target images (green or blue)
  3. Overlays method paths from all method images (red)
  4. Overlays start markers (red or yellow) from target images
  5. Overlays goal markers (blue or green) from target images
  6. Saves the result as a single combined image

Path inflation:
  - Makes paths thicker/more visible on the final image
  - Inflates by N pixels in all directions (circular)
  - Example: --target-inflation 2 makes paths 2 pixels wider on each side

Color options:
  --yellow-start : Start marker red → yellow
  --blue-target  : Target path green → blue
  --green-goal   : Goal marker blue → green

Expected inputs:
  - background: Original image (e.g., the map/scene)
  - targets: Images with green paths and markers (from TIFF editor)
  - methods: Images with red paths (resized method outputs)
  
All images should have the same dimensions.
        """
    )
    
    parser.add_argument('background', help='Background image (original scene)')
    parser.add_argument('targets_and_methods', nargs='+', metavar='image',
                       help='Target images followed by method images (use --num-methods to specify how many are methods)')
    parser.add_argument('--num-methods', type=int, default=1, metavar='N',
                       help='Number of method images (taken from end of image list, default: 1)')
    parser.add_argument('-o', '--output', help='Output filename (default: background_combined.ext)')
    parser.add_argument('--target-inflation', type=int, default=0, metavar='N',
                       help='Inflate target paths by N pixels for visibility (default: 0)')
    parser.add_argument('--method-inflation', type=int, default=0, metavar='N',
                       help='Inflate method paths by N pixels for visibility (default: 0)')
    parser.add_argument('--yellow-start', action='store_true',
                       help='Change start marker color from red to yellow')
    parser.add_argument('--blue-target', action='store_true',
                       help='Change target path color from green to blue')
    parser.add_argument('--green-goal', action='store_true',
                       help='Change goal marker color from blue to green')
    
    args = parser.parse_args()
    
    # Parse targets and methods from the combined list
    # Last num_methods images are methods, rest are targets
    all_images = args.targets_and_methods
    num_methods = args.num_methods
    
    if num_methods >= len(all_images):
        print(f"Error: --num-methods ({num_methods}) must be less than total images ({len(all_images)})")
        sys.exit(1)
    
    background = args.background
    # Split into targets and methods
    targets = all_images[:-num_methods] if num_methods > 0 else all_images
    methods = all_images[-num_methods:] if num_methods > 0 else []
    
    if len(targets) == 0:
        print("Error: At least one target image is required")
        sys.exit(1)
    
    if len(methods) == 0:
        print("Error: At least one method image is required")
        sys.exit(1)
    
    try:
        combiner = PathOverlayCombiner(
            background_path=background,
            target_paths=targets,
            method_paths=methods,
            target_inflation=args.target_inflation,
            method_inflation=args.method_inflation,
            yellow_start=args.yellow_start,
            blue_target=args.blue_target,
            green_goal=args.green_goal
        )
        
        output_path = combiner.save_combined(args.output)
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Combined overlay image created: {output_path}")
        print("\nYou can now view all paths and markers together on the original background.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
