"""Main script for generating standalone Voronoi decompositions.

This script creates three types of Voronoi visualizations:
1. Black and white line drawings
2. Colored diagrams with cells colored by perimeter
3. Random walk visualizations with multiple color schemes

The generated SVG files are saved with timestamped names in the project root.
"""

from typing import Dict, Tuple
import drawing


def main():
    """Generate various Voronoi diagram visualizations."""
    
    # Configuration parameters
    NUMBER_OF_CELLS = 3000
    
    # Viewport bounds (10x10 coordinate system)
    BOUNDS = {
        "xmin": 0, 
        "ymin": 0, 
        "xmax": 10, 
        "ymax": 10
    }
    
    # Perimeter range for color mapping
    PERIMETER_BOUNDS = {
        "max": 1.6, 
        "min": 0.2
    }
    
    # Color schemes for different visualizations
    # Each defines a gradient from start to end color
    COLOR_SCHEMES = {
        "red_to_green": {"start": (255, 0, 0), "end": (8, 255, 0)},
        "blue_to_yellow": {"start": (0, 90, 255), "end": (80, 255, 0)},
        "red_to_brown": {"start": (255, 0, 0), "end": (120, 50, 0)},
        "red_to_lime": {"start": (255, 0, 0), "end": (120, 240, 0)}
    }
    
    # Generate black and white diagram
    print("Generating black and white Voronoi diagram...")
    drawing.black_and_white('bw', NUMBER_OF_CELLS)
    
    # Generate colored diagram
    print("Generating colored Voronoi diagram...")
    drawing.color('color', NUMBER_OF_CELLS, BOUNDS, PERIMETER_BOUNDS, 
                 COLOR_SCHEMES["red_to_green"])
    
    # Generate batch of random walk visualizations
    print("Generating random walk visualizations...")
    drawing.random_walk_drawings(
        'rw', 
        20,  # Generate 20 different random walk images
        NUMBER_OF_CELLS, 
        BOUNDS, 
        PERIMETER_BOUNDS,
        COLOR_SCHEMES["red_to_green"],
        COLOR_SCHEMES["blue_to_yellow"],
        COLOR_SCHEMES["red_to_brown"],
        COLOR_SCHEMES["red_to_lime"]
    )
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()