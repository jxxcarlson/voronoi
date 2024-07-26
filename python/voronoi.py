# Produce SVG files for Voronoi decompositions
# Key polygon color to the perimeter of the polygon.

from lib import *
import drawing

bounds = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
perimeter_bounds = {"max": 1.6, "min": 0.2}

color_bounds1 = {"start": (255, 0, 0), "end": (8, 255, 0) }
color_bounds2 = {"start": (0, 90, 255), "end": (80, 255, 0) }
color_bounds3 = {"start": (255, 0, 0), "end": (120, 50, 0) }
color_bounds4 = {"start": (255, 0, 0), "end": (120, 240, 0) }

# statistics()


drawing.black_and_white('bw', 3000)
drawing.color('color', 3000, bounds, perimeter_bounds, color_bounds1)
drawing.random_walk_drawing('rw' , 3000, bounds, perimeter_bounds, color_bounds1, color_bounds2, color_bounds3, color_bounds4 )