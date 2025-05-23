"""Convert images to Voronoi mosaic visualizations.

This module processes PNG images to create artistic Voronoi mosaics where:
- Points are distributed based on image brightness/color values
- Voronoi cells are colored based on the original image colors
- The resulting mosaic maintains the essence of the original image

The process involves:
1. Converting the image to HSB (Hue, Saturation, Brightness) color space
2. Generating random points with probability based on brightness
3. Creating a Voronoi diagram from these points
4. Coloring cells based on sampled colors from the original image
"""

import numpy as np
from PIL import Image
import colorsys
from scipy.spatial import Voronoi
import svgwrite
import cairosvg
import argparse
import os
from typing import Tuple, List, Optional


def png_to_hsb_matrix(file_path: str) -> np.ndarray:
    """Convert a PNG image to an HSB color matrix.
    
    Args:
        file_path: Path to the PNG image file
        
    Returns:
        3D numpy array of shape (height, width, 3) containing HSB values
    """
    # Open and convert to RGB
    img = Image.open(file_path).convert('RGB')
    width, height = img.size
    
    # Initialize HSB matrix
    hsb_matrix = np.zeros((height, width, 3))
    
    # Convert each pixel from RGB to HSB
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsb_matrix[y, x] = (h, s, v)
    
    return hsb_matrix


def chunk_hsb_matrix(hsb_matrix: np.ndarray, chunk_size: int) -> np.ndarray:
    """Downsample an HSB matrix by averaging chunks of pixels.
    
    Args:
        hsb_matrix: Original HSB matrix
        chunk_size: Size of chunks to average (chunk_size x chunk_size)
        
    Returns:
        Downsampled HSB matrix
    """
    height, width, _ = hsb_matrix.shape
    chunked_height = height // chunk_size
    chunked_width = width // chunk_size
    
    chunked_hsb_matrix = np.zeros((chunked_height, chunked_width, 3))
    
    # Average each chunk
    for y in range(chunked_height):
        for x in range(chunked_width):
            chunk = hsb_matrix[y*chunk_size:(y+1)*chunk_size, 
                              x*chunk_size:(x+1)*chunk_size]
            chunked_hsb_matrix[y, x] = np.mean(chunk, axis=(0, 1))
    
    return chunked_hsb_matrix


def hsb_matrix_to_png(hsb_matrix: np.ndarray, output_path: str, 
                     upscale_factor: int = 1) -> None:
    """Convert an HSB matrix back to a PNG image.
    
    Args:
        hsb_matrix: HSB color matrix
        output_path: Path to save the PNG file
        upscale_factor: Factor to upscale the image
    """
    height, width, _ = hsb_matrix.shape
    img = Image.new('RGB', (width * upscale_factor, height * upscale_factor))
    
    for y in range(height):
        for x in range(width):
            h, s, v = hsb_matrix[y, x]
            r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
            # Fill upscaled pixels
            for i in range(upscale_factor):
                for j in range(upscale_factor):
                    img.putpixel((x * upscale_factor + i, y * upscale_factor + j), 
                               (r, g, b))
    
    img.save(output_path)
    print(f"Image saved to {output_path}")


def svg_to_png(svg_path: str, png_path: str, 
              width: Optional[int] = None, height: Optional[int] = None) -> None:
    """Convert an SVG file to a PNG file using CairoSVG.
    
    Args:
        svg_path: Path to the input SVG file
        png_path: Path to save the output PNG file
        width: Desired width of the PNG (optional)
        height: Desired height of the PNG (optional)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    # Read SVG file
    with open(svg_path, 'rb') as svg_file:
        svg_data = svg_file.read()
    
    # Prepare conversion arguments
    kwargs = {}
    if width is not None:
        kwargs['output_width'] = int(width)
    if height is not None:
        kwargs['output_height'] = int(height)
    
    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_data, **kwargs)
    
    # Write PNG file
    with open(png_path, 'wb') as png_file:
        png_file.write(png_data)
    
    print(f"PNG image saved as {png_path}")


def generate_random_points_with_color(hsb_matrix: np.ndarray, N: int, size: float,
                                    f=lambda x: x) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random points based on image brightness with color data.
    
    Points are generated with probability proportional to the brightness
    value at their location in the image.
    
    Args:
        hsb_matrix: HSB color matrix from the image
        N: Number of candidate points to generate
        size: Width of the output canvas
        f: Function to transform brightness to probability (default: identity)
        
    Returns:
        Tuple of (points array, points with color data array)
    """
    rows, columns, _ = hsb_matrix.shape
    aspect_ratio = columns / rows
    
    # Calculate canvas dimensions
    width = size
    height = size / aspect_ratio
    
    # Lists for accepted points
    RP = []  # Random points
    RP_Colors = []  # Points with color data
    
    for _ in range(N):
        # Generate random candidate point
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        # Find corresponding pixel in HSB matrix
        row = int(y / height * rows)
        col = int(x / width * columns)
        
        # Get HSB values and calculate acceptance probability
        h, s, v = hsb_matrix[row, col]
        p = f(v)  # Probability based on brightness
        
        # Accept point with probability p
        if np.random.random() < p:
            RP.append([x, y])
            RP_Colors.append([x, y, h, s, v])
    
    return np.array(RP), np.array(RP_Colors)


def plot_voronoi_color(points: np.ndarray, colors: np.ndarray, size: float,
                      aspect_ratio: float, svg_output_path: str, 
                      png_output_path: str) -> None:
    """Create a colored Voronoi mosaic and save as SVG and PNG.
    
    Args:
        points: Array of point coordinates
        colors: Array of points with HSB color data
        size: Canvas width in pixels
        aspect_ratio: Width/height ratio
        svg_output_path: Path to save SVG file
        png_output_path: Path to save PNG file
    """
    # Compute Voronoi tessellation
    vor = Voronoi(points)
    
    # Create SVG drawing
    dwg = svgwrite.Drawing(svg_output_path, 
                          size=(f'{size}px', f'{size/aspect_ratio}px'))
    
    # White background
    dwg.add(dwg.rect(insert=(0, 0), 
                    size=(f'{size}px', f'{size/aspect_ratio}px'), 
                    fill='white'))
    
    # Map points to their colors
    point_to_color = {tuple(point[:2]): color[2:5] 
                     for point, color in zip(points, colors)}
    
    # Track region colors for edge coloring
    region_colors = {}
    
    # Draw colored Voronoi cells
    print(f"Drawing {len(vor.regions)} Voronoi cells...")
    bbox = (0, 0, size, size/aspect_ratio)
    
    for i, region in enumerate(vor.regions):
        if not -1 in region and len(region) > 0:
            polygon = vor.vertices[region]
            # Check if polygon is within bounds
            if (len(polygon) > 2 and 
                all((bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]) 
                    for x, y in polygon)):
                
                # Find the point for this region
                region_point = vor.points[vor.point_region == i][0]
                h, s, v = point_to_color.get(tuple(region_point), (0, 0, 1))
                
                # Store color for edge drawing
                region_colors[i] = (h, s, v)
                
                # Convert HSV to RGB
                r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
                color = svgwrite.rgb(r, g, b)
                dwg.add(dwg.polygon(points=polygon, fill=color, 
                                   stroke=color, stroke_width=0.5))
    
    # Draw Voronoi edges with colors matching regions
    for simplex, ridge_points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in simplex:
            start = vor.vertices[simplex[0]]
            end = vor.vertices[simplex[1]]
            if all((bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]) 
                   for x, y in (start, end)):
                # Use color from first region
                region1, region2 = ridge_points
                h, s, v = region_colors.get(vor.point_region[region1], (0, 0, 1))
                r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, 1)]
                edge_color = svgwrite.rgb(r, g, b)
                dwg.add(dwg.line(start=start, end=end, 
                               stroke=edge_color, stroke_width=0.5))
    
    # Save SVG
    dwg.save()
    print(f"Voronoi SVG saved as {svg_output_path}")
    
    # Convert to PNG
    svg_to_png(svg_output_path, png_output_path, 
              width=size, height=int(size/aspect_ratio))


def get_file_name(file_path: str) -> str:
    """Extract filename without extension from a path.
    
    Args:
        file_path: Full file path
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def main(file_path: str, n_voronoi_cells: int, chunk_size: int) -> None:
    """Generate a Voronoi mosaic from an input image.
    
    Args:
        file_path: Path to input PNG image
        n_voronoi_cells: Target number of Voronoi cells
        chunk_size: Downsampling chunk size (currently unused)
    """
    # Validate input file
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return
    
    # Setup output directory
    file_name = get_file_name(file_path)
    image_out_prefix = 'images_out'
    os.makedirs(image_out_prefix, exist_ok=True)
    
    # Convert image to HSB
    print(f"Loading image: {file_path}")
    hsb_matrix = png_to_hsb_matrix(file_path)
    print(f"Image dimensions: {hsb_matrix.shape[1]}x{hsb_matrix.shape[0]}")
    
    # Generate random points based on image brightness
    print(f"Generating up to {n_voronoi_cells} Voronoi points...")
    size = 1000  # Canvas width in pixels
    random_points, random_points_colors = generate_random_points_with_color(
        hsb_matrix, n_voronoi_cells, size, lambda x: x
    )
    print(f"Generated {len(random_points)} points based on image brightness")
    
    # Calculate aspect ratio
    aspect_ratio = hsb_matrix.shape[1] / hsb_matrix.shape[0]
    
    # Create and save colored Voronoi mosaic
    print("Creating colored Voronoi mosaic...")
    svg_output_path = f"{image_out_prefix}/{file_name}_voronoi_color.svg"
    png_output_path = f"{image_out_prefix}/{file_name}_voronoi_color.png"
    plot_voronoi_color(random_points, random_points_colors, size, aspect_ratio,
                      svg_output_path, png_output_path)
    
    print(f"\nMosaic generation complete!")
    print(f"Output files:\n  - {svg_output_path}\n  - {png_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Voronoi mosaics from images.",
        epilog="Example: python mosaic_from_image.py image.png 16000 1"
    )
    parser.add_argument("file_path", help="Path to the input PNG image")
    parser.add_argument("n_voronoi_cells", type=int, 
                       help="Target number of Voronoi cells")
    parser.add_argument("chunk_size", type=int, 
                       help="Chunk size for downsampling (currently unused)")
    
    args = parser.parse_args()
    main(args.file_path, args.n_voronoi_cells, args.chunk_size)