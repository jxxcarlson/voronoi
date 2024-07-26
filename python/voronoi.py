# Produce SVG files for Voronoi decompositions
# Key polygon color to the perimeter of the polygon.



import numpy as np
from scipy.spatial import Voronoi
import svgwrite
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random
import math
from lib import *
import drawing

# Define the visible rectangle
xmin, ymin, xmax, ymax = 0, 0, 10, 10

# Generate random points



# HELPER FUNCTIONS




# voronoi_polygons = get_voronoi_polygons(vor)
drawing.black_and_white('bw', 3000)
# print(f"Number of Voronoi polygons: {len(voronoi_polygons)}")







# drawing = svgwrite.Drawing('voronoi_color.svg', size=('1000px', '1000px'), viewBox='0 0 10 10')

#  COLOR SVG FILE


# ... (previous code remains the same)


start_color = (255, 0, 0)  # Red
end_color = (80, 255, 0)    # Blue

start_color2 = (0, 90, 255)  # Red
end_color2 = (0, 255, 0)    # Blue

start_color3 = (255, 0, 0)  # Red
end_color3 = (120, 50, 0)    # Blue

start_color4 = (255, 0, 0)  # Red
end_color4 = (120, 2400, 0)    # Blue

# Calculate perimeters for all polygons
# perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]



min_perimeter, max_perimeter = 0.2, 1.6 #min(perimeters), max(perimeters)




# Continue with the rest of your code...

# def clip(value, min_value, max_value):
#     return max(min(value, max_value), min_value)



bounds = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
perimeter_bounds = {"max": 1.6, "min": 0.2}

color_bounds1 = {"start": (255, 0, 0), "end": (8, 255, 0) }
color_bounds2 = {"start": (0, 90, 255), "end": (80, 255, 0) }
color_bounds3 = {"start": (255, 0, 0), "end": (120, 50, 0) }
color_bounds4 = {"start": (255, 0, 0), "end": (120, 240, 0) }
# statistics()
# color_drawing('color')
# drawing.black_and_white('bw', 3000)
# drawing.color('color', 3000, bounds, perimeter_bounds, color_bounds1)
drawing.random_walk_drawing('rw' , 3000, bounds, perimeter_bounds, color_bounds1, color_bounds2, color_bounds3, color_bounds4 )