"""Utility script for analyzing HSB color values in images and generating test visualizations.

This module provides tools to:
- Convert images to HSB (Hue, Saturation, Brightness) color space
- Downsample images by chunking and averaging
- Generate Voronoi diagrams based on image brightness
- Create test visualizations for development

This is primarily a development/testing utility demonstrating the core
image processing techniques used in mosaic_from_image.py.
"""

import numpy as np
from PIL import Image
import colorsys
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import svgwrite
import os
from typing import Tuple, Optional


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


def generate_random_points(hsb_matrix: np.ndarray, N: int, size: float) -> np.ndarray:
    """Generate random points based on image brightness.
    
    Points are generated with probability equal to the brightness value
    at their location in the image.
    
    Args:
        hsb_matrix: HSB color matrix from the image
        N: Number of candidate points to generate
        size: Width of the output canvas
        
    Returns:
        Array of accepted point coordinates
    """
    rows, columns, _ = hsb_matrix.shape
    aspect_ratio = columns / rows
    
    # Calculate canvas dimensions
    width = size
    height = size / aspect_ratio
    
    # List for accepted points
    RP = []
    
    for _ in range(N):
        # Generate random candidate point
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        # Find corresponding pixel in HSB matrix
        row = int(y / height * rows)
        col = int(x / width * columns)
        
        # Get brightness value as probability
        _, _, v = hsb_matrix[row, col]
        p = v
        
        # Accept point with probability p
        if np.random.random() < p:
            RP.append([x, y])
    
    return np.array(RP)


def voronoi_to_svg(points: np.ndarray, size: float, aspect_ratio: float, 
                  output_path: str) -> None:
    """Create a Voronoi diagram and save as SVG.
    
    Args:
        points: Array of point coordinates
        size: Canvas width
        aspect_ratio: Width/height ratio
        output_path: Path to save SVG file
    """
    # Compute Voronoi tessellation
    vor = Voronoi(points)
    
    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, 
                          size=(f'{size}px', f'{size/aspect_ratio}px'))
    
    # Draw Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:  # Skip infinite edges
            start = vor.vertices[simplex[0]]
            end = vor.vertices[simplex[1]]
            dwg.add(dwg.line(start=start, end=end, 
                           stroke='black', stroke_width=0.5))
    
    # Save SVG
    dwg.save()
    print(f"Voronoi SVG saved as {output_path}")


def plot_voronoi(points: np.ndarray, size: float, aspect_ratio: float,
                output_path: Optional[str] = None) -> None:
    """Plot a Voronoi diagram using matplotlib.
    
    Args:
        points: Array of point coordinates
        size: Canvas width
        aspect_ratio: Width/height ratio
        output_path: Optional path to save the plot as PNG
    """
    # Compute Voronoi tessellation
    vor = Voronoi(points)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 
                    'k-', linewidth=0.5)
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], s=1)
    
    # Set limits and aspect ratio
    plt.xlim(0, size)
    plt.ylim(0, size / aspect_ratio)
    ax.set_aspect('equal', adjustable='box')
    
    # Remove axes
    plt.axis('off')
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Voronoi plot saved as {output_path}")
    
    # Display the plot
    plt.show()


def main():
    """Run example visualization using a test image."""
    # Configuration
    INPUT_IMAGE = 'images_in/klee.png'
    OUTPUT_DIR = 'images_out'
    CHUNK_SIZE = 128
    NUM_POINTS = 1000
    CANVAS_SIZE = 1000
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process image
    print(f"Loading image: {INPUT_IMAGE}")
    hsb_matrix = png_to_hsb_matrix(INPUT_IMAGE)
    print(f"Original image shape: {hsb_matrix.shape}")
    
    # Chunk the image
    print(f"Chunking with size {CHUNK_SIZE}x{CHUNK_SIZE}")
    chunked_hsb_matrix = chunk_hsb_matrix(hsb_matrix, CHUNK_SIZE)
    print(f"Chunked image shape: {chunked_hsb_matrix.shape}")
    
    # Save chunked image
    chunked_output = f'{OUTPUT_DIR}/chunked_klee.png'
    hsb_matrix_to_png(chunked_hsb_matrix, chunked_output, upscale_factor=CHUNK_SIZE)
    
    # Generate random points based on brightness
    print(f"Generating up to {NUM_POINTS} random points...")
    random_points = generate_random_points(chunked_hsb_matrix, NUM_POINTS, CANVAS_SIZE)
    print(f"Generated {len(random_points)} points")
    
    # Calculate aspect ratio
    aspect_ratio = chunked_hsb_matrix.shape[1] / chunked_hsb_matrix.shape[0]
    
    # Create and save Voronoi diagram
    print("Creating Voronoi diagram...")
    voronoi_to_svg(random_points, CANVAS_SIZE, aspect_ratio, 
                  f'{OUTPUT_DIR}/voronoi_diagram.svg')
    
    plot_voronoi(random_points, CANVAS_SIZE, aspect_ratio,
                f'{OUTPUT_DIR}/voronoi_plot.png')
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()