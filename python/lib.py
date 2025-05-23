"""Core utility functions for Voronoi diagram generation and manipulation.

This module provides fundamental algorithms and utilities for:
- Timestamp-based file naming
- Polygon clipping and manipulation
- Voronoi cell generation with custom point distributions
- Random walk algorithms through Voronoi cells
- Statistical analysis of polygon properties
"""

from datetime import datetime
from scipy.spatial import Voronoi, cKDTree
import time
from collections import defaultdict
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Set, Optional


# FILE NAMING UTILITIES

def get_timestamp() -> str:
    """Generate a timestamp string for unique file naming.
    
    Returns:
        str: Timestamp in format 'YYYY.MM.DD-HH.MM.SS'
    """
    now = datetime.now()
    return now.strftime("%Y.%m.%d-%H.%M.%S")


def make_file_name(prefix: str) -> str:
    """Create a timestamped SVG filename.
    
    Args:
        prefix: Base name for the file
        
    Returns:
        str: Filename in format 'prefix_timestamp.svg'
    """
    return f"{prefix}_{get_timestamp()}.svg"


def make_numbered_file_name(prefix: str, number: int) -> str:
    """Create a numbered and timestamped SVG filename.
    
    Args:
        prefix: Base name for the file
        number: Number to include in filename
        
    Returns:
        str: Filename in format 'prefix_number_timestamp.svg'
    """
    return f"{prefix}_{number}_{get_timestamp()}.svg"


# POLYGON MANIPULATION

def clip_polygon(poly: List[Tuple[float, float]], xmin: float, ymin: float, 
                xmax: float, ymax: float) -> List[Tuple[float, float]]:
    """Clip a polygon to a rectangular boundary using Cohen-Sutherland algorithm.
    
    Args:
        poly: List of (x, y) vertices defining the polygon
        xmin, ymin: Bottom-left corner of clipping rectangle
        xmax, ymax: Top-right corner of clipping rectangle
        
    Returns:
        List of (x, y) vertices of the clipped polygon
    """
    def clip_line(x1: float, y1: float, x2: float, y2: float) -> List[Tuple[float, float]]:
        """Clip a line segment to the boundary rectangle."""
        code1 = encode(x1, y1)
        code2 = encode(x2, y2)
        accept = False
        
        while True:
            if not (code1 | code2):  # Both endpoints inside
                accept = True
                break
            elif code1 & code2:  # Both endpoints outside same region
                break
            else:  # Line potentially crosses boundary
                x, y = 0.0, 0.0
                code_out = code1 if code1 else code2
                
                if code_out & 1:  # Left of rectangle
                    x = xmin
                    y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                elif code_out & 2:  # Right of rectangle
                    x = xmax
                    y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                elif code_out & 4:  # Below rectangle
                    y = ymin
                    x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                elif code_out & 8:  # Above rectangle
                    y = ymax
                    x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                    
                if code_out == code1:
                    x1, y1 = x, y
                    code1 = encode(x1, y1)
                else:
                    x2, y2 = x, y
                    code2 = encode(x2, y2)
                    
        return [(x1, y1), (x2, y2)] if accept else []

    def encode(x: float, y: float) -> int:
        """Encode point position relative to clipping rectangle."""
        code = 0
        if x < xmin:
            code |= 1
        elif x > xmax:
            code |= 2
        if y < ymin:
            code |= 4
        elif y > ymax:
            code |= 8
        return code

    clipped_poly = []
    for i in range(len(poly)):
        line = clip_line(poly[i][0], poly[i][1], 
                        poly[(i+1) % len(poly)][0], poly[(i+1) % len(poly)][1])
        clipped_poly.extend(line)
    return clipped_poly


def polygon_perimeter(poly: List[Tuple[float, float]]) -> float:
    """Calculate the perimeter of a polygon.
    
    Args:
        poly: List of (x, y) vertices defining the polygon
        
    Returns:
        float: Total perimeter length
    """
    return sum(math.sqrt((x2-x1)**2 + (y2-y1)**2) 
               for (x1, y1), (x2, y2) in zip(poly, poly[1:] + [poly[0]]))


def interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                     factor: float) -> Tuple[int, int, int]:
    """Linearly interpolate between two RGB colors.
    
    Args:
        color1: Starting RGB color (0-255 per channel)
        color2: Ending RGB color (0-255 per channel)
        factor: Interpolation factor (0.0 = color1, 1.0 = color2)
        
    Returns:
        Tuple of interpolated RGB values
    """
    return tuple(int(color1[i] + (color2[i] - color1[i]) * factor) for i in range(3))


def clip(value: float, min_value: float, max_value: float) -> float:
    """Constrain a value to be within a given range.
    
    Args:
        value: Value to constrain
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        float: Constrained value
    """
    return max(min(value, max_value), min_value)


# VORONOI CELL GENERATION

def get_voronoi_polygons(vor: Voronoi) -> List[List[Tuple[float, float]]]:
    """Extract valid polygon regions from a Voronoi diagram.
    
    Args:
        vor: scipy.spatial.Voronoi object
        
    Returns:
        List of polygons, where each polygon is a list of (x, y) vertices
    """
    polygons = []
    for region in vor.regions:
        # Skip empty regions and regions with infinite vertices
        if len(region) > 0 and all(i != -1 for i in region):
            polygon = [vor.vertices[i] for i in region]
            polygons.append(polygon)
    return polygons


def _point_probability(x: float, y: float) -> float:
    """Calculate probability of keeping a point based on its position.
    
    Creates a non-uniform distribution with higher density in certain regions.
    
    Args:
        x, y: Point coordinates
        
    Returns:
        float: Probability of keeping the point (0.0 to 1.0)
    """
    if x - y < -1 or x < 2:
        return 1.0
    if x - y < 2 or y < 2:
        return 0.5
    else:
        return 0.25


def voronoiCells(num_points: int) -> Voronoi:
    """Generate a Voronoi diagram with non-uniform point distribution.
    
    Creates a set of points with density varying based on position,
    then computes the Voronoi diagram.
    
    Args:
        num_points: Target number of points (actual may be less due to filtering)
        
    Returns:
        scipy.spatial.Voronoi object
    """
    # Generate random candidate points in [0, 10) x [0, 10)
    points_ = np.random.rand(num_points, 2) * 10
        
    # Apply probability mask to create non-uniform distribution
    mask = np.random.random(len(points_)) < np.array([_point_probability(x, y) 
                                                      for x, y in points_])
    
    # Filter points based on probability
    points = points_[mask]
    
    print(f"Generated {len(points_)} candidates, kept {len(points)} "
          f"(fraction = {len(points)/len(points_):.3f})")
    
    # Compute and return Voronoi diagram
    return Voronoi(points)


# RANDOM WALK ALGORITHMS

def random_walk(voronoi_polygons: List[List[Tuple[float, float]]], 
               steps: int, timeout: int = 30) -> List[int]:
    """Perform a random walk through neighboring Voronoi cells.
    
    Starting from a random cell, walks to adjacent cells by randomly
    selecting shared edges.
    
    Args:
        voronoi_polygons: List of polygons defining Voronoi cells
        steps: Maximum number of steps to take
        timeout: Maximum time in seconds before stopping
        
    Returns:
        List of polygon indices visited during the walk
    """
    start_time = time.time()
    neighbors = find_neighboring_polygons(voronoi_polygons)
    
    # Start from random polygon
    current = random.randint(0, len(voronoi_polygons) - 1)
    walk = [current]
    
    for _ in range(steps - 1):
        # Check timeout
        if time.time() - start_time > timeout:
            print(f"Random walk timed out after {timeout} seconds. "
                  f"Returning partial walk of {len(walk)} steps.")
            break
            
        # Move to random neighbor if available
        if neighbors[current]:
            current = random.choice(list(neighbors[current]))
            walk.append(current)
        else:
            break  # No neighbors available
    
    return walk


def find_neighboring_polygons(voronoi_polygons: List[List[Tuple[float, float]]], 
                            tolerance: float = 1e-6) -> Dict[int, Set[int]]:
    """Find adjacency relationships between Voronoi polygons.
    
    Two polygons are neighbors if they share an edge (within tolerance).
    Uses spatial indexing for efficient edge matching.
    
    Args:
        voronoi_polygons: List of polygons to analyze
        tolerance: Maximum distance for edges to be considered shared
        
    Returns:
        Dictionary mapping polygon index to set of neighbor indices
    """
    neighbors = defaultdict(set)
    all_edges = []
    edge_centers = []
    
    # Extract all edges and their centers
    for i, poly in enumerate(voronoi_polygons):
        for j in range(len(poly)):
            edge = (tuple(poly[j]), tuple(poly[(j+1) % len(poly)]))
            all_edges.append((i, edge))
            edge_centers.append(np.mean(edge, axis=0))
    
    edge_centers = np.array(edge_centers)
    if edge_centers.size == 0:
        print("Warning: No edges found. Check if voronoi_polygons is empty.")
        return neighbors

    # Build spatial index for efficient neighbor search
    tree = cKDTree(edge_centers)
    
    # Find matching edges
    for i, (poly_idx, edge) in enumerate(all_edges):
        edge_center = np.mean(edge, axis=0)
        potential_neighbors = tree.query_ball_point(edge_center, r=tolerance)
        
        for j in potential_neighbors:
            if i != j:
                other_poly_idx, other_edge = all_edges[j]
                # Check if edges match (considering reversed orientation)
                if (np.allclose(edge[0], other_edge[0], atol=tolerance) and 
                    np.allclose(edge[1], other_edge[1], atol=tolerance)) or \
                   (np.allclose(edge[0], other_edge[1], atol=tolerance) and 
                    np.allclose(edge[1], other_edge[0], atol=tolerance)):
                    neighbors[poly_idx].add(other_poly_idx)
                    neighbors[other_poly_idx].add(poly_idx)
    
    return neighbors


# STATISTICAL ANALYSIS

def statistics(data: List[float]) -> None:
    """Print statistical summary of numerical data.
    
    Displays minimum, maximum, median, and percentiles of the data,
    formatted to 4 significant figures.
    
    Args:
        data: List of numerical values to analyze
    """
    def format_4sig(x: float) -> str:
        """Format number to 4 significant figures."""
        return f"{x:.4g}"

    sorted_data = sorted(data)

    # Basic statistics
    print(f"Minimum: {format_4sig(sorted_data[0])}")
    print(f"Maximum: {format_4sig(sorted_data[-1])}")
    print(f"Median: {format_4sig(sorted_data[len(sorted_data)//2])}")

    # Percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"{p:3d}th: {format_4sig(np.percentile(sorted_data, p))}")