"""SVG drawing functions for creating Voronoi diagrams with various visual styles.

This module provides functions to render Voronoi diagrams as SVG files with:
- Black and white line drawings
- Colored cells based on polygon perimeter
- Random walk visualizations through colored cells
- Multiple color schemes and visual effects
"""

from typing import List, Tuple, Dict, Optional
import svgwrite
import os
import webbrowser

from lib import (
    make_file_name, make_numbered_file_name, voronoiCells, 
    clip_polygon, interpolate_color, get_voronoi_polygons, 
    polygon_perimeter, random_walk
)


def black_and_white(filename_prefix: str, number_of_cells: int) -> None:
    """Create a black and white Voronoi diagram showing cell edges.
    
    Args:
        filename_prefix: Base name for the output file
        number_of_cells: Target number of Voronoi cells to generate
    """
    filename = make_file_name(filename_prefix)
    vor = voronoiCells(number_of_cells)
    
    # Create SVG with 10x10 coordinate system
    drawing = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')
    
    # Draw Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:  # Skip infinite edges
            p1, p2 = vor.vertices[simplex]
            drawing.add(drawing.line(start=p1, end=p2, stroke='black', stroke_width=0.02))
    
    print(filename)
    drawing.save()


def draw_with_color(voronoi_polygons: List[List[Tuple[float, float]]], 
                   drawing: svgwrite.Drawing,
                   bounds: Dict[str, float], 
                   perimeter_bounds: Dict[str, float],
                   color_bounds: Dict[str, Tuple[int, int, int]]) -> None:
    """Draw colored Voronoi cells with color based on perimeter.
    
    Args:
        voronoi_polygons: List of polygons to draw
        drawing: SVG drawing object to add elements to
        bounds: Dictionary with 'xmin', 'ymin', 'xmax', 'ymax' for clipping
        perimeter_bounds: Dictionary with 'min' and 'max' perimeter values
        color_bounds: Dictionary with 'start' and 'end' RGB colors
    """
    for poly in voronoi_polygons:
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], 
                                   bounds["xmax"], bounds["ymax"])
        if clipped_poly:  # Only draw if polygon is visible
            # Calculate color based on perimeter
            perimeter = polygon_perimeter(poly)
            factor = (perimeter - perimeter_bounds["min"]) / (perimeter_bounds["max"] - perimeter_bounds["min"])
            color = interpolate_color(color_bounds["start"], color_bounds["end"], factor)
            
            drawing.add(drawing.polygon(
                points=clipped_poly,
                fill=svgwrite.rgb(*color, '%'),
                fill_opacity=1.0,
                stroke='blue',
                stroke_width=0.01,
                stroke_opacity=0.5
            ))


def draw_with_color_rw(walk_steps: int,
                      voronoi_polygons: List[List[Tuple[float, float]]],
                      perimeters: List[float],
                      drawing: svgwrite.Drawing,
                      bounds: Dict[str, float],
                      perimeter_bounds: Dict[str, float],
                      color_bounds: Dict[str, Tuple[int, int, int]]) -> svgwrite.Drawing:
    """Draw a random walk through Voronoi cells with colored polygons.
    
    Args:
        walk_steps: Number of steps to take in the random walk
        voronoi_polygons: List of all Voronoi polygons
        perimeters: Pre-calculated perimeters for each polygon
        drawing: SVG drawing to add elements to
        bounds: Clipping bounds
        perimeter_bounds: Range of perimeter values for color mapping
        color_bounds: Start and end colors for interpolation
        
    Returns:
        Modified drawing object
    """
    random_walk_indices = random_walk(voronoi_polygons, walk_steps, timeout=60)
    
    for i, index in enumerate(random_walk_indices):
        poly = voronoi_polygons[index]
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], 
                                   bounds["xmax"], bounds["ymax"])
        if clipped_poly:
            # Color based on perimeter
            perimeter = perimeters[index]
            factor = (perimeter - perimeter_bounds["min"]) / (perimeter_bounds["max"] - perimeter_bounds["min"])
            color = interpolate_color(color_bounds["start"], color_bounds["end"], factor)
            
            drawing.add(drawing.polygon(
                points=clipped_poly,
                fill=svgwrite.rgb(*color, '%'),
                fill_opacity=0.45,
                stroke='black',
                stroke_width=0.02
            ))
    
    return drawing


def color(filename_prefix: str, 
         number_of_cells: int,
         bounds: Dict[str, float],
         perimeter_bounds: Dict[str, float],
         color_bounds: Dict[str, Tuple[int, int, int]]) -> None:
    """Create a colored Voronoi diagram with cells colored by perimeter.
    
    Args:
        filename_prefix: Base name for output file
        number_of_cells: Target number of Voronoi cells
        bounds: Viewport bounds for clipping
        perimeter_bounds: Range of perimeter values for color mapping
        color_bounds: Start and end colors for interpolation
    """
    filename = make_file_name(filename_prefix)
    drawing = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')
    
    # White background
    drawing.add(drawing.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))
    
    # Generate Voronoi diagram
    vor = voronoiCells(number_of_cells)
    voronoi_polygons = get_voronoi_polygons(vor)
    
    # Draw colored cells
    draw_with_color(voronoi_polygons, drawing, bounds, perimeter_bounds, color_bounds)
    
    print(filename)
    drawing.save()


def random_walk_drawing(filename_prefix: str,
                       number_of_cells: int,
                       bounds: Dict[str, float],
                       perimeter_bounds: Dict[str, float],
                       color_bounds1: Dict[str, Tuple[int, int, int]],
                       color_bounds2: Dict[str, Tuple[int, int, int]],
                       color_bounds3: Dict[str, Tuple[int, int, int]],
                       color_bounds4: Dict[str, Tuple[int, int, int]]) -> None:
    """Create a Voronoi diagram with multiple random walks in different colors.
    
    Creates a dark background with black cells, then overlays multiple
    random walks using different color schemes.
    
    Args:
        filename_prefix: Base name for output file
        number_of_cells: Target number of Voronoi cells
        bounds: Viewport bounds for clipping
        perimeter_bounds: Range of perimeter values for color mapping
        color_bounds1-4: Different color schemes for each random walk
    """
    filename = make_file_name(filename_prefix)
    drawing_walk = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')
    
    # Forest green background
    drawing_walk.add(drawing_walk.rect(insert=(0, 0), size=('100%', '100%'), fill='forestgreen'))
    
    # Generate Voronoi diagram
    vor = voronoiCells(number_of_cells)
    voronoi_polygons = get_voronoi_polygons(vor)
    perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]
    
    # Draw all polygons in black
    for poly in voronoi_polygons:
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], 
                                   bounds["xmax"], bounds["ymax"])
        if clipped_poly:
            drawing_walk.add(drawing_walk.polygon(
                points=clipped_poly,
                fill='black',
                fill_opacity=0.9,
                stroke='rgb(40, 40, 40)',
                stroke_width=0.02
            ))
    
    # Draw multiple random walks with different colors
    walk_steps = 1400
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds1)
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds2)
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds3)
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds4)
    
    print(filename)
    drawing_walk.save()
    
    # Open in browser
    html_filename = filename.replace('.svg', '.html')
    with open(html_filename, 'w') as f:
        f.write(f'<html><body><img src="{os.path.basename(filename)}" /></body></html>')
    
    webbrowser.open('file://' + os.path.realpath(html_filename))


def random_walk_drawing_(filename_prefix: str,
                        number: int,
                        number_of_cells: int,
                        bounds: Dict[str, float],
                        perimeter_bounds: Dict[str, float],
                        color_bounds1: Dict[str, Tuple[int, int, int]],
                        color_bounds2: Dict[str, Tuple[int, int, int]],
                        color_bounds3: Dict[str, Tuple[int, int, int]],
                        color_bounds4: Dict[str, Tuple[int, int, int]]) -> None:
    """Create a numbered random walk drawing (without browser opening).
    
    Similar to random_walk_drawing but with numbered filename and no browser launch.
    Used for batch generation.
    
    Args:
        filename_prefix: Base name for output file
        number: Number to include in filename
        number_of_cells: Target number of Voronoi cells
        bounds: Viewport bounds for clipping
        perimeter_bounds: Range of perimeter values for color mapping
        color_bounds1-4: Different color schemes for each random walk
    """
    filename = make_numbered_file_name(filename_prefix, number)
    drawing_walk = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')
    
    # Forest green background
    drawing_walk.add(drawing_walk.rect(insert=(0, 0), size=('100%', '100%'), fill='forestgreen'))
    
    # Generate Voronoi diagram
    vor = voronoiCells(number_of_cells)
    voronoi_polygons = get_voronoi_polygons(vor)
    perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]
    
    # Draw all polygons in black
    for poly in voronoi_polygons:
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], 
                                   bounds["xmax"], bounds["ymax"])
        if clipped_poly:
            drawing_walk.add(drawing_walk.polygon(
                points=clipped_poly,
                fill='black',
                fill_opacity=0.9,
                stroke='rgb(40, 40, 40)',
                stroke_width=0.02
            ))
    
    # Draw multiple random walks
    walk_steps = 1400
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds1)
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds2)
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds3)
    draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, 
                      bounds, perimeter_bounds, color_bounds4)
    
    print(filename)
    drawing_walk.save()


def random_walk_drawings(filename_prefix: str,
                        count: int,
                        number_of_cells: int,
                        bounds: Dict[str, float],
                        perimeter_bounds: Dict[str, float],
                        color_bounds1: Dict[str, Tuple[int, int, int]],
                        color_bounds2: Dict[str, Tuple[int, int, int]],
                        color_bounds3: Dict[str, Tuple[int, int, int]],
                        color_bounds4: Dict[str, Tuple[int, int, int]]) -> None:
    """Generate multiple random walk drawings in batch.
    
    Args:
        filename_prefix: Base name for output files
        count: Number of drawings to generate
        number_of_cells: Target number of Voronoi cells per drawing
        bounds: Viewport bounds for clipping
        perimeter_bounds: Range of perimeter values for color mapping
        color_bounds1-4: Different color schemes for each random walk
    """
    for k in range(count):
        random_walk_drawing_(filename_prefix, k, number_of_cells, bounds, 
                           perimeter_bounds, color_bounds1, color_bounds2, 
                           color_bounds3, color_bounds4)