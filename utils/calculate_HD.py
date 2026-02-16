#!/usr/bin/env python3
"""
Path Comparison using Hausdorff Distance
Extracts pure green path from target image and pure red path from method image,
then calculates the unified directed Hausdorff distance between them.
"""

import numpy as np
from PIL import Image
import sys
import argparse
from pathlib import Path


def unified_directed_hausdorff_distance(target_masks, method_mask):
    """
    Calculate the unified directed Hausdorff distance from method to target masks.
    
    Args:
        target_masks: List of binary masks (numpy arrays) from target image(s)
        method_mask: Single binary mask (numpy array) from method image
        
    Returns:
        float: The unified directed Hausdorff distance
    """
    # Get coordinates of all points in method mask
    method_points = np.argwhere(method_mask > 0)
    
    if len(method_points) == 0:
        print("Warning: Method mask is empty (no red path found)")
        return float('inf')
    
    # Collect all target points from all target masks
    target_points = []
    for target_mask in target_masks:
        mask_points = np.argwhere(target_mask > 0)
        if len(mask_points) > 0:
            target_points.append(mask_points)
    
    if len(target_points) == 0:
        print("Warning: All target masks are empty (no green path found)")
        return float('inf')
    
    # Concatenate all target points
    all_target_points = np.vstack(target_points)
    
    print(f"Method points: {len(method_points)}")
    print(f"Target points: {len(all_target_points)}")
    
    # Calculate Hausdorff distance
    # For each point in method, find minimum distance to any target point
    max_min_distance = 0.0
    
    for method_point in method_points:
        # Calculate distances from this method point to all target points
        distances = np.sqrt(np.sum((all_target_points - method_point) ** 2, axis=1))
        min_distance = np.min(distances)
        
        # Keep track of the maximum of these minimum distances
        if min_distance > max_min_distance:
            max_min_distance = min_distance
    
    return max_min_distance


class PathComparator:
    def __init__(self, target_paths, method_path):
        """
        Initialize the path comparator.
        
        Args:
            target_paths: List of paths to target images (with pure green paths)
            method_path: Path to method image (with pure red path)
        """
        # Convert to list if single path provided
        if isinstance(target_paths, (str, Path)):
            target_paths = [target_paths]
        
        self.target_paths = [Path(p) for p in target_paths]
        self.method_path = Path(method_path)
        
        # Verify all target files exist
        for target_path in self.target_paths:
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
        
        if not self.method_path.exists():
            raise FileNotFoundError(f"Method image not found: {method_path}")
        
        # Load target images
        self.target_imgs = []
        self.target_arrays = []
        
        print(f"Loading {len(self.target_paths)} target image(s):")
        for i, target_path in enumerate(self.target_paths, 1):
            target_img = Image.open(target_path)
            target_array = np.array(target_img.convert('RGB'))
            self.target_imgs.append(target_img)
            self.target_arrays.append(target_array)
            print(f"  {i}. {target_path.name}: {target_array.shape}")
        
        # Load method image
        self.method_img = Image.open(self.method_path)
        self.method_array = np.array(self.method_img.convert('RGB'))
        
        print(f"\nMethod image loaded: {self.method_array.shape}")
        
        # Verify dimensions match (all targets should match method)
        method_shape = self.method_array.shape
        for i, target_array in enumerate(self.target_arrays, 1):
            if target_array.shape != method_shape:
                print(f"\nWARNING: Target {i} dimensions don't match method!")
                print(f"  Target {i}: {target_array.shape}")
                print(f"  Method: {method_shape}")
                print(f"  Please ensure all images were resized to match.")
    
    def extract_green_path_masks(self):
        """
        Extract binary masks of pure green (0, 255, 0) pixels from all target images.
        
        Returns:
            list: List of binary masks where green path pixels = 1, others = 0
        """
        # Pure green is RGB(0, 255, 0)
        pure_green = np.array([0, 255, 0])
        
        masks = []
        total_green_pixels = 0
        
        print(f"\nTarget images (green paths):")
        
        for i, target_array in enumerate(self.target_arrays, 1):
            # Create binary mask where all three channels match pure green
            mask = np.all(target_array == pure_green, axis=2).astype(np.uint8)
            green_pixel_count = np.sum(mask)
            total_green_pixels += green_pixel_count
            
            print(f"  Target {i}: {green_pixel_count} pure green pixels")
            
            if green_pixel_count == 0:
                print(f"    WARNING: No pure green (0,255,0) pixels found in target {i}!")
            
            masks.append(mask)
        
        print(f"  Total green pixels across all targets: {total_green_pixels}")
        
        if total_green_pixels == 0:
            print("  WARNING: No pure green pixels found in any target image!")
            print("  Make sure the target images have green paths drawn on them.")
        
        return masks
    
    def extract_red_path_mask(self):
        """
        Extract binary mask of pure red (255, 0, 0) pixels from method image.
        
        Returns:
            numpy.ndarray: Binary mask where red path pixels = 1, others = 0
        """
        # Pure red is RGB(255, 0, 0)
        pure_red = np.array([255, 0, 0])
        
        # Create binary mask where all three channels match pure red
        mask = np.all(self.method_array == pure_red, axis=2).astype(np.uint8)
        
        red_pixel_count = np.sum(mask)
        print(f"\nMethod image (red path):")
        print(f"  Pure red pixels found: {red_pixel_count}")
        
        if red_pixel_count == 0:
            print("  WARNING: No pure red (255,0,0) pixels found in method image!")
            print("  Make sure the method image has a red path drawn on it.")
        
        return mask
    
    def calculate_hausdorff_distance(self):
        """
        Calculate the unified directed Hausdorff distance between paths.
        
        Returns:
            float: The Hausdorff distance
        """
        print("\n" + "="*70)
        print("EXTRACTING PATHS")
        print("="*70)
        
        # Extract masks
        target_masks = self.extract_green_path_masks()
        method_mask = self.extract_red_path_mask()
        
        # Check if masks have pixels
        total_target_pixels = sum(np.sum(mask) for mask in target_masks)
        
        if total_target_pixels == 0 or np.sum(method_mask) == 0:
            print("\n" + "="*70)
            print("ERROR: Cannot calculate distance - one or both paths are empty")
            print("="*70)
            return None
        
        print("\n" + "="*70)
        print("CALCULATING HAUSDORFF DISTANCE")
        print("="*70)
        
        # Calculate distance (target_masks is already a list as required)
        distance = unified_directed_hausdorff_distance(target_masks, method_mask)
        
        return distance
    
    def save_masks(self, output_dir=None):
        """
        Save the extracted masks as images for visualization.
        
        Args:
            output_dir: Directory to save masks (default: same as first target image)
        """
        if output_dir is None:
            output_dir = self.target_paths[0].parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract masks
        target_masks = self.extract_green_path_masks()
        method_mask = self.extract_red_path_mask()
        
        saved_paths = []
        
        # Save each target mask
        for i, (target_path, target_mask) in enumerate(zip(self.target_paths, target_masks), 1):
            # Convert mask to image (0 -> black, 1 -> white)
            target_mask_img = Image.fromarray((target_mask * 255).astype(np.uint8), mode='L')
            
            # Create filename with index if multiple targets
            if len(self.target_paths) > 1:
                target_mask_path = output_dir / f"{target_path.stem}_green_mask_{i}.png"
            else:
                target_mask_path = output_dir / f"{target_path.stem}_green_mask.png"
            
            target_mask_img.save(target_mask_path)
            print(f"\n✓ Saved target {i} mask (green path) to: {target_mask_path}")
            saved_paths.append(target_mask_path)
        
        # Save method mask
        method_mask_img = Image.fromarray((method_mask * 255).astype(np.uint8), mode='L')
        method_mask_path = output_dir / f"{self.method_path.stem}_red_mask.png"
        method_mask_img.save(method_mask_path)
        
        print(f"✓ Saved method mask (red path) to: {method_mask_path}")
        saved_paths.append(method_mask_path)
        
        return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description='Compare paths using Hausdorff distance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target comparison
  python path_comparator.py target_with_green_path.png method_resized_with_red_path.png
  
  # Multiple targets comparison
  python path_comparator.py target1.png target2.png target3.png method_resized.png
  
  # Save extracted masks for visualization
  python path_comparator.py target.png method.png --save-masks
  
  # Multiple targets with mask saving
  python path_comparator.py target1.png target2.png method.png --save-masks --output-dir ./masks

Expected input:
  - Target image(s): Should contain pure green RGB(0,255,0) paths
  - Method image: Should contain a pure red RGB(255,0,0) path
  - All images should be the same dimensions
  
The program will:
  1. Extract green paths from all target images as binary masks
  2. Extract red path from method image as binary mask
  3. Calculate unified directed Hausdorff distance
  4. Print the result
        """
    )
    
    parser.add_argument('targets', nargs='+', metavar='target', 
                       help='One or more target images with pure green (0,255,0) paths')
    parser.add_argument('method', help='Method image with pure red (255,0,0) path')
    parser.add_argument('--save-masks', action='store_true',
                       help='Save extracted binary masks as images')
    parser.add_argument('--output-dir', help='Directory to save masks (default: same as input)')
    
    args = parser.parse_args()
    
    try:
        # Create comparator with multiple targets
        comparator = PathComparator(
            target_paths=args.targets,
            method_path=args.method
        )
        
        # Calculate Hausdorff distance
        distance = comparator.calculate_hausdorff_distance()
        
        if distance is not None:
            print("\n" + "="*70)
            print("RESULT")
            print("="*70)
            print(f"\nUnified Directed Hausdorff Distance: {distance:.4f} pixels")
            print()
            print("Interpretation:")
            print(f"  - The maximum minimum distance from any red path pixel")
            print(f"    to the nearest green path pixel is {distance:.4f} pixels.")
            print(f"  - Lower values indicate better path matching.")
            print(f"  - A value of 0 means paths overlap perfectly.")
            print("="*70)
            
            # Save masks if requested
            if args.save_masks:
                print()
                comparator.save_masks(args.output_dir)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
