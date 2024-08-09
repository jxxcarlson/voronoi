from lib import make_file_name, make_numbered_file_name, voronoiCells, clip_polygon, interpolate_color, get_voronoi_polygons, polygon_perimeter, random_walk
import svgwrite
import os
import webbrowser

def black_and_white(filename_, number_of_cells):

    filename = make_file_name(filename_)

    vor = voronoiCells(number_of_cells)

    # Create black and white SVG drawing
    drawing = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')

    # Draw Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            p1, p2 = vor.vertices[simplex]
            drawing.add(drawing.line(start=p1, end=p2, stroke='black', stroke_width=0.02))

    print(filename)
    drawing.save()


def draw_with_color(voronoi_polygons, drawing, bounds, perimeter_bounds, color_bounds):
     for poly in voronoi_polygons:
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"])
        if clipped_poly:  # Only draw the polygon if it's not completely clipped
            perimeter = polygon_perimeter(poly)
            factor = (perimeter - perimeter_bounds["min"]) / (perimeter_bounds["max"] - perimeter_bounds["min"])
            color = interpolate_color(color_bounds["start"], color_bounds["end"], factor)
            drawing.add(drawing.polygon(points=clipped_poly, 
                            fill=svgwrite.rgb(*color, '%'), 
                            fill_opacity=1.0, 
                            stroke='blue', 
                            stroke_width=0.01,
                            stroke_opacity=0.5))
            
def draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing, bounds, perimeter_bounds, color_bounds):
    random_walk_indices = random_walk(voronoi_polygons, walk_steps, timeout=60) 
    for i, index in enumerate(random_walk_indices):
        poly = voronoi_polygons[index]
        clipped_poly = clip_polygon(poly,bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"])
        if clipped_poly:
            # color = interpolate_color((255, 0, 0), (100, 255, 0), i / (len(random_walk_indices) - 1))
            perimeter = perimeters[index]
            factor = (perimeter - perimeter_bounds["min"]) / (perimeter_bounds["max"] - perimeter_bounds["min"])
            color = interpolate_color(color_bounds["start"], color_bounds["end"], factor)
            drawing.add(drawing.polygon(points=clipped_poly,                                       
                                                fill=svgwrite.rgb(*color, '%'), 
                                                fill_opacity=0.45,
                                                stroke='black', 
                                                stroke_width=0.02))
    return drawing

def color(filename_,  number_of_cells, bounds, perimeter_bounds, color_bounds):

    filename = make_file_name(filename_)

    drawing = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')

    # Add a white background
    drawing.add(drawing.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    vor = voronoiCells(number_of_cells)

    voronoi_polygons = get_voronoi_polygons(vor) 
    perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]

    draw_with_color(voronoi_polygons, drawing, bounds, perimeter_bounds, color_bounds)

    print(filename)
    drawing.save()


def random_walk_drawing(filename_,  number_of_cells, bounds, perimeter_bounds, color_bounds1, color_bounds2, color_bounds3, color_bounds4):

    filename = make_file_name(filename_)

    drawing_walk = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')

    # Add a white background
    drawing_walk.add(drawing_walk.rect(insert=(0, 0), size=('100%', '100%'), fill='forestgreen'))


    vor = voronoiCells(number_of_cells)

    voronoi_polygons = get_voronoi_polygons(vor) 
    perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]
    # Draw all polygons in light gray
    for poly in voronoi_polygons:
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"])        
        if clipped_poly:
            drawing_walk.add(drawing_walk.polygon(points=clipped_poly, 
                                                # fill='rgb(40, 160, 40)',  
                                                fill = 'black',
                                                fill_opacity=0.9,
                                                stroke='rgb(40, 40, 40)', 
                                                stroke_width=0.02))

    # When calling random_walk, you can specify a timeout:
    walk_steps = 1400
    random_walk_indices = random_walk(voronoi_polygons, walk_steps, timeout=60)  # 60-second timeout
    random_walk_indices2 = random_walk(voronoi_polygons, walk_steps, timeout=60)  # 60-second timeout
    # Draw the random walk polygons
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds1)
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds2)
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds3)
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds4)                                                  

    print(filename)

    svg_filename = drawing_walk.filename
    drawing_walk.save()

    chrome_path = 'open -a /Applications/Firefox.app %s'
    
    html_filename = svg_filename.replace('.svg', '.html')
    with open(html_filename, 'w') as f:
        f.write(f'<html><body><img src="{os.path.basename(svg_filename)}" /></body></html>')

    # Open the HTML file in the default browser
    webbrowser.open('file://' + os.path.realpath(html_filename))
    # webbrowser.open('file://' + os.path.realpath(drawing_walk.filename))


def random_walk_drawing_(filename_, k, number_of_cells, bounds, perimeter_bounds, color_bounds1, color_bounds2, color_bounds3, color_bounds4):

    filename = make_numbered_file_name(filename_, k)

    drawing_walk = svgwrite.Drawing(filename, size=('1000px', '1000px'), viewBox='0 0 10 10')

    # Add a white background
    drawing_walk.add(drawing_walk.rect(insert=(0, 0), size=('100%', '100%'), fill='forestgreen'))


    vor = voronoiCells(number_of_cells)

    voronoi_polygons = get_voronoi_polygons(vor) 
    perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]
    # Draw all polygons in light gray
    for poly in voronoi_polygons:
        clipped_poly = clip_polygon(poly, bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"])        
        if clipped_poly:
            drawing_walk.add(drawing_walk.polygon(points=clipped_poly, 
                                                # fill='rgb(40, 160, 40)',  
                                                fill = 'black',
                                                fill_opacity=0.9,
                                                stroke='rgb(40, 40, 40)', 
                                                stroke_width=0.02))

    # When calling random_walk, you can specify a timeout:
    walk_steps = 1400
    random_walk_indices = random_walk(voronoi_polygons, walk_steps, timeout=60)  # 60-second timeout
    random_walk_indices2 = random_walk(voronoi_polygons, walk_steps, timeout=60)  # 60-second timeout
    # Draw the random walk polygons
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds1)
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds2)
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds3)
    drawing_walk = draw_with_color_rw(walk_steps, voronoi_polygons, perimeters, drawing_walk, bounds, perimeter_bounds, color_bounds4)                                                  

    print(filename)

    drawing_walk.save()


def random_walk_drawings(filename_, n, number_of_cells, bounds, perimeter_bounds, color_bounds1, color_bounds2, color_bounds3, color_bounds4):
    for k in range(0,n):
        random_walk_drawing_(filename_, k, number_of_cells, bounds, perimeter_bounds, color_bounds1, color_bounds2, color_bounds3, color_bounds4)
    
