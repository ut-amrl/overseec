#!/usr/bin/env python3
"""
Interactive TIFF Editor with Multi-layer Support
Allows marking start/goal locations and drawing paths on TIFF images.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Configure matplotlib backend BEFORE importing pyplot
import matplotlib
# Try to use an interactive backend
backends_to_try = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'WXAgg', 'MacOSX']
backend_set = False

for backend in backends_to_try:
    try:
        matplotlib.use(backend, force=True)
        backend_set = True
        print(f"Using matplotlib backend: {backend}")
        break
    except:
        continue

if not backend_set:
    print("WARNING: Could not set an interactive backend.")
    print("Available backends:", matplotlib.rcsetup.all_backends)
    print("Trying default backend...")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image


class TiffEditor:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load the original image
        self.original_img = Image.open(self.filepath)
        
        # Extract RGB layer (first layer)
        self.rgb_layer = np.array(self.original_img.convert('RGB'))
        self.height, self.width = self.rgb_layer.shape[:2]
        
        # Create working copy for display
        self.display_img = self.rgb_layer.copy()
        
        # Initialize marker layers (same size as image)
        self.start_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.goal_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.path_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Store marker positions
        self.start_pos = None
        self.goal_pos = None
        self.path_points = []
        
        # Marker properties (configurable)
        self.ring_radius = 40       # Radius of the ring in pixels
        self.dot_radius = 12         # Radius of the center dot in pixels
        self.ring_thickness = 8     # Thickness of the ring line in pixels
        self.path_line_width = 6    # Width of the path line in pixels
        
        # Mouse tracking for path drawing
        self.is_drawing = False
        
    def draw_ring_marker(self, layer, position, color):
        """Draw a ring with a dot at center on the specified layer.
        The ring and dot sizes match what's displayed in the GUI.
        """
        y, x = position
        y, x = int(y), int(x)
        
        # Draw center dot (filled circle)
        for dy in range(-self.dot_radius, self.dot_radius + 1):
            for dx in range(-self.dot_radius, self.dot_radius + 1):
                if dx*dx + dy*dy <= self.dot_radius*self.dot_radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        layer[ny, nx] = color
        
        # Draw thick ring (hollow circle with specified thickness)
        # We draw multiple concentric circles to achieve thickness
        for thickness_offset in range(self.ring_thickness):
            current_radius = self.ring_radius - thickness_offset / 2
            num_points = int(2 * np.pi * current_radius * 2)  # More points for smoother circle
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                ring_y = int(y + current_radius * np.sin(angle))
                ring_x = int(x + current_radius * np.cos(angle))
                
                if 0 <= ring_y < self.height and 0 <= ring_x < self.width:
                    layer[ring_y, ring_x] = color
                    
                    # Add thickness by drawing adjacent pixels
                    if thickness_offset == 0:
                        for dy in range(-self.ring_thickness // 2, self.ring_thickness // 2 + 1):
                            for dx in range(-self.ring_thickness // 2, self.ring_thickness // 2 + 1):
                                ny, nx = ring_y + dy, ring_x + dx
                                if 0 <= ny < self.height and 0 <= nx < self.width:
                                    # Only draw if we're within the ring thickness
                                    dist_from_center = np.sqrt((nx - x)**2 + (ny - y)**2)
                                    if abs(dist_from_center - self.ring_radius) <= self.ring_thickness / 2:
                                        layer[ny, nx] = color
    
    def mark_start_location(self):
        """Interactive marking of start location."""
        print("\n=== MARKING START LOCATION ===")
        print("Click on the image to mark the START location (pure red)")
        print("Close the window when done.")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(self.display_img)
        ax.set_title("Click to mark START location (Red)", fontsize=14)
        ax.axis('off')
        
        def onclick(event):
            if event.inaxes == ax and event.button == 1:  # Left click
                x, y = int(event.xdata), int(event.ydata)
                self.start_pos = (y, x)
                
                # Draw on start layer
                self.start_layer.fill(0)  # Clear previous
                self.draw_ring_marker(self.start_layer, (y, x), 
                                     color=[255, 0, 0])  # Pure red
                
                # Update display
                temp_display = self.rgb_layer.copy()
                # Overlay start marker
                mask = np.any(self.start_layer > 0, axis=2)
                temp_display[mask] = self.start_layer[mask]
                
                ax.clear()
                ax.imshow(temp_display)
                ax.set_title(f"START marked at ({x}, {y})", fontsize=14)
                ax.axis('off')
                
                # Draw the ring and dot for visualization (matching the actual output)
                circle_ring = Circle((x, y), self.ring_radius, 
                                   fill=False, color='red', linewidth=self.ring_thickness)
                circle_dot = Circle((x, y), self.dot_radius, 
                                  fill=True, color='red')
                ax.add_patch(circle_ring)
                ax.add_patch(circle_dot)
                
                fig.canvas.draw()
                print(f"Start location marked at pixel ({x}, {y})")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()
        
        if self.start_pos is None:
            print("Warning: No start location was marked!")
    
    def mark_goal_location(self):
        """Interactive marking of goal location."""
        print("\n=== MARKING GOAL LOCATION ===")
        print("Click on the image to mark the GOAL location (pure blue)")
        print("Close the window when done.")
        
        # Update display with start marker if it exists
        temp_display = self.rgb_layer.copy()
        if self.start_pos is not None:
            mask = np.any(self.start_layer > 0, axis=2)
            temp_display[mask] = self.start_layer[mask]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(temp_display)
        ax.set_title("Click to mark GOAL location (Blue)", fontsize=14)
        ax.axis('off')
        
        def onclick(event):
            if event.inaxes == ax and event.button == 1:  # Left click
                x, y = int(event.xdata), int(event.ydata)
                self.goal_pos = (y, x)
                
                # Draw on goal layer
                self.goal_layer.fill(0)  # Clear previous
                self.draw_ring_marker(self.goal_layer, (y, x), 
                                     color=[0, 0, 255])  # Pure blue
                
                # Update display
                temp_display = self.rgb_layer.copy()
                # Overlay start and goal markers
                start_mask = np.any(self.start_layer > 0, axis=2)
                goal_mask = np.any(self.goal_layer > 0, axis=2)
                temp_display[start_mask] = self.start_layer[start_mask]
                temp_display[goal_mask] = self.goal_layer[goal_mask]
                
                ax.clear()
                ax.imshow(temp_display)
                ax.set_title(f"GOAL marked at ({x}, {y})", fontsize=14)
                ax.axis('off')
                
                # Draw the ring and dot for visualization (matching the actual output)
                if self.start_pos is not None:
                    sx, sy = self.start_pos[1], self.start_pos[0]
                    circle_ring = Circle((sx, sy), self.ring_radius, 
                                       fill=False, color='red', linewidth=self.ring_thickness)
                    circle_dot = Circle((sx, sy), self.dot_radius, 
                                      fill=True, color='red')
                    ax.add_patch(circle_ring)
                    ax.add_patch(circle_dot)
                
                circle_ring = Circle((x, y), self.ring_radius, 
                                   fill=False, color='blue', linewidth=self.ring_thickness)
                circle_dot = Circle((x, y), self.dot_radius, 
                                  fill=True, color='blue')
                ax.add_patch(circle_ring)
                ax.add_patch(circle_dot)
                
                fig.canvas.draw()
                print(f"Goal location marked at pixel ({x}, {y})")
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()
        
        if self.goal_pos is None:
            print("Warning: No goal location was marked!")
    
    def save_with_markers(self):
        """Save the image with start and goal markers to _edit_1.tiff"""
        output_path = self.filepath.parent / f"{self.filepath.stem}_edit_1.tiff"
        
        # Create composite image with markers
        result = self.rgb_layer.copy()
        
        # Overlay markers (goal last so it's on top if overlapping)
        start_mask = np.any(self.start_layer > 0, axis=2)
        goal_mask = np.any(self.goal_layer > 0, axis=2)
        
        result[start_mask] = self.start_layer[start_mask]
        result[goal_mask] = self.goal_layer[goal_mask]
        
        # Convert to PIL Image and save
        img = Image.fromarray(result, mode='RGB')
        img.save(output_path, compression='tiff_deflate')
        
        print(f"\n✓ Saved image with markers to: {output_path}")
        return output_path
    
    def draw_path(self):
        """Interactive path drawing with mouse."""
        print("\n=== DRAWING PATH ===")
        print("Click and drag to draw a path (pure green)")
        print("Release mouse button to stop drawing")
        print("Click and drag again to continue drawing")
        print("Close the window when done.")
        
        # Update display with markers
        temp_display = self.rgb_layer.copy()
        start_mask = np.any(self.start_layer > 0, axis=2)
        goal_mask = np.any(self.goal_layer > 0, axis=2)
        temp_display[start_mask] = self.start_layer[start_mask]
        temp_display[goal_mask] = self.goal_layer[goal_mask]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(temp_display)
        ax.set_title("Click and drag to draw path (Green). Close window when done.", 
                    fontsize=14)
        ax.axis('off')
        
        self.path_points = []
        last_point = None
        
        def on_press(event):
            if event.inaxes == ax and event.button == 1:
                self.is_drawing = True
        
        def on_release(event):
            if event.button == 1:
                self.is_drawing = False
                nonlocal last_point
                last_point = None
        
        def on_motion(event):
            nonlocal last_point
            if self.is_drawing and event.inaxes == ax:
                x, y = int(event.xdata), int(event.ydata)
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Draw line from last point to current point for smooth lines
                    if last_point is not None:
                        x0, y0 = last_point
                        # Bresenham's line algorithm for smooth drawing
                        points = self._get_line_points(x0, y0, x, y)
                        for px, py in points:
                            if 0 <= px < self.width and 0 <= py < self.height:
                                # Draw with thickness
                                self._draw_thick_point(px, py, [0, 255, 0], self.path_line_width)
                                self.path_points.append((px, py))
                    else:
                        # Draw with thickness
                        self._draw_thick_point(x, y, [0, 255, 0], self.path_line_width)
                        self.path_points.append((x, y))
                    
                    last_point = (x, y)
                    
                    # Update display
                    temp_display = self.rgb_layer.copy()
                    temp_display[start_mask] = self.start_layer[start_mask]
                    temp_display[goal_mask] = self.goal_layer[goal_mask]
                    path_mask = np.any(self.path_layer > 0, axis=2)
                    temp_display[path_mask] = self.path_layer[path_mask]
                    
                    ax.clear()
                    ax.imshow(temp_display)
                    ax.set_title(f"Drawing path... ({len(self.path_points)} points)", 
                               fontsize=14)
                    ax.axis('off')
                    fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Path drawing complete: {len(self.path_points)} points")
    
    def _draw_thick_point(self, x, y, color, thickness):
        """Draw a point with specified thickness (width)."""
        half_thickness = thickness // 2
        for dy in range(-half_thickness, half_thickness + 1):
            for dx in range(-half_thickness, half_thickness + 1):
                # Draw a square brush (for faster rendering)
                # Could use circular brush: if dx*dx + dy*dy <= (thickness/2)**2:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.path_layer[ny, nx] = color
    
    def _get_line_points(self, x0, y0, x1, y1):
        """Bresenham's line algorithm to get all points between two points."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def save_with_path(self):
        """Save the image with path to _user_path.tiff (WITHOUT start/goal markers)"""
        output_path = self.filepath.parent / f"{self.filepath.stem}_user_path.tiff"
        
        # Create composite image with ONLY the path (no markers)
        result = self.rgb_layer.copy()
        
        # Overlay only the path layer
        path_mask = np.any(self.path_layer > 0, axis=2)
        result[path_mask] = self.path_layer[path_mask]
        
        # Convert to PIL Image and save
        img = Image.fromarray(result, mode='RGB')
        img.save(output_path, compression='tiff_deflate')
        
        print(f"\n✓ Saved image with path (no markers) to: {output_path}")
        return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python tiff_editor.py <input_tiff_file>")
        print("Example: python tiff_editor.py image.tiff")
        sys.exit(1)
    
    # Check if matplotlib backend is interactive
    backend = matplotlib.get_backend()
    print(f"Current matplotlib backend: {backend}")
    
    if not matplotlib.rcsetup.interactive_bk or backend == 'agg':
        print("\n" + "="*60)
        print("ERROR: No interactive display backend available!")
        print("="*60)
        print("\nThis program requires a GUI display to work.")
        print("\nPossible solutions:")
        print("1. If on Linux, install tkinter:")
        print("   Ubuntu/Debian: sudo apt-get install python3-tk")
        print("   Fedora/RHEL: sudo dnf install python3-tkinter")
        print("   Arch: sudo pacman -S tk")
        print("\n2. If using SSH, enable X11 forwarding:")
        print("   ssh -X user@host")
        print("\n3. Try installing PyQt5:")
        print("   pip install PyQt5")
        print("\n4. If on WSL, install an X server (VcXsrv, Xming)")
        print("="*60)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        print(f"Loading TIFF file: {input_file}")
        editor = TiffEditor(input_file)
        
        print(f"Image dimensions: {editor.width} x {editor.height}")
        
        # Step 1: Mark start location
        editor.mark_start_location()
        
        # Step 2: Mark goal location
        editor.mark_goal_location()
        
        # Step 3: Save with markers
        if editor.start_pos or editor.goal_pos:
            editor.save_with_markers()
        else:
            print("\nNo markers were placed, skipping _edit_1.tiff")
        
        # Step 4: Draw path
        editor.draw_path()
        
        # Step 5: Save with path
        if editor.path_points:
            editor.save_with_path()
        else:
            print("\nNo path was drawn, skipping _user_path.tiff")
        
        print("\n" + "="*50)
        print("Processing complete!")
        print("Original file unchanged.")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
