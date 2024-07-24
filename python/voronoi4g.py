# Produce SVG files for Voronoi decompositions
# Key polygon color to the perimeter of the polygon.


suffix = '_4g'
filename1 = 'voronoi_line' + suffix + '.svg'
filename2 = 'voronoi_color' + suffix + '.svg'

import numpy as np
from scipy.spatial import Voronoi
import svgwrite
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random
import math


# Generate random points
num_points = 2560  # You can adjust this number
points_ = np.random.rand(num_points, 2) * 10  # Create 2D array of points and scale the coordinates to [0, 10).


# Probability of creating a point at (x,y)
def p(x,y):
    if x - y < -1 or x < 2:
        return 1.0
    if x - y < 2 or y < 2:
        return 0.5
    else:
        return 0.25
    
# Create a mask based on the probability function:
# np.array([p(x, y) for x, y in points_]) creates 
# an array of probabilities by applying p(x,y) to
# points_ .
mask = np.random.random(len(points_)) < np.array([p(x, y) for x, y in points_])

# Apply the mask to points_ to create points
points = points_[mask]

print("len(points_) = ", len(points_), "len(points) = ", len(points), "fraction = {:.3f}".format(len(points)/len(points_)))

# Compute the Voronoi diagram using scipy.spatial
vor = Voronoi(points)


# HELPER FUNCTIONS


def clip_polygon(poly, xmin, ymin, xmax, ymax):
    def clip_line(x1, y1, x2, y2):
        code1 = encode(x1, y1)
        code2 = encode(x2, y2)
        accept = False
        while True:
            if not (code1 | code2):  # Both endpoints inside the clip rectangle
                accept = True
                break
            elif code1 & code2:  # Both endpoints are outside the clip rectangle
                break
            else:  # Line potentially crosses the clip rectangle
                x, y = 0, 0
                code_out = code1 if code1 else code2
                if code_out & 1:  # Point is to the left of clip rectangle
                    x = xmin
                    y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                elif code_out & 2:  # Point is to the right of clip rectangle
                    x = xmax
                    y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                elif code_out & 4:  # Point is below the clip rectangle
                    y = ymin
                    x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                elif code_out & 8:  # Point is above the clip rectangle
                    y = ymax
                    x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                if code_out == code1:
                    x1, y1 = x, y
                    code1 = encode(x1, y1)
                else:
                    x2, y2 = x, y
                    code2 = encode(x2, y2)
        if accept:
            return [(x1, y1), (x2, y2)]
        else:
            return []

    def encode(x, y):
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
        line = clip_line(poly[i][0], poly[i][1], poly[(i+1)%len(poly)][0], poly[(i+1)%len(poly)][1])
        clipped_poly.extend(line)
    return clipped_poly



def get_voronoi_polygons(vor):
    polygons = []
    for region in vor.regions:
        if len(region) > 0 and all(i != -1 for i in region):
            polygon = [vor.vertices[i] for i in region]
            polygons.append(polygon)
    return polygons







voronoi_polygons = get_voronoi_polygons(vor)
print(f"Number of Voronoi polygons: {len(voronoi_polygons)}")


# Create black and white SVG drawing
drawing = svgwrite.Drawing(filename1, size=('1000px', '1000px'), viewBox='0 0 10 10')

# Draw Voronoi edges
for simplex in vor.ridge_vertices:
    if -1 not in simplex:
        p1, p2 = vor.vertices[simplex]
        drawing.add(drawing.line(start=p1, end=p2, stroke='black', stroke_width=0.02))

# Draw points
# for point in points:
#     drawing.add(drawing.circle(center=point, r=0.1, fill='red'))

# Add title
# drawing.add(drawing.text('Voronoi Diagram with Random Points', insert=(5, 0.5), 
#                  text_anchor='middle', font_size='0.5px', fill='black'))

# Save the SVG file
drawing.save()



# drawing = svgwrite.Drawing('voronoi_color.svg', size=('1000px', '1000px'), viewBox='0 0 10 10')

#  COLOR SVG FILE

drawing = svgwrite.Drawing(filename2, size=('1000px', '1000px'), viewBox='0 0 10 10')


# Add a white background
drawing.add(drawing.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))


# ... (previous code remains the same)

def polygon_perimeter(poly):
    return sum(math.sqrt((x2-x1)**2 + (y2-y1)**2) for (x1, y1), (x2, y2) in zip(poly, poly[1:] + [poly[0]]))

def interpolate_color(color1, color2, factor):
    return tuple(int(color1[i] + (color2[i] - color1[i]) * factor) for i in range(3))

def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

start_color = (255, 0, 0)  # Red
end_color = (80, 255, 0)    # Blue

# Calculate perimeters for all polygons
perimeters = [polygon_perimeter(poly) for poly in voronoi_polygons]


sorted_perimeters= sorted(perimeters)

# Function to format number to 4 significant figures
def format_4sig(x):
    return f"{x:.4g}"



# Print the min, max, and median of N
print(f"Minimum perimeter: {format_4sig(sorted_perimeters[0])}")
print(f"Maximum perimeter: {format_4sig(sorted_perimeters[-1])}")
print(f"Median perimeter: {format_4sig(sorted_perimeters[len(sorted_perimeters)//2])}")

# Print percentiles
percentiles = [0, 10, 25, 50, 75, 90, 100]
print("Len percentiles:")
for p in percentiles:
    print(f"{p}th percentile: {format_4sig(np.percentile(sorted_perimeters, p))}")

min_perimeter, max_perimeter = 0.6, 1.6 #min(perimeters), max(perimeters)

# Define the visible rectangle
xmin, ymin, xmax, ymax = 0, 0, 10, 10

drawing = svgwrite.Drawing(filename2, size=('1000px', '1000px'), viewBox='0 0 10 10')

# Add a white background
drawing.add(drawing.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))


for poly, perimeter in zip(voronoi_polygons, perimeters):
    clipped_poly = clip_polygon(poly, xmin, ymin, xmax, ymax)
    if clipped_poly:  # Only draw the polygon if it's not completely clipped
        factor = (perimeter - min_perimeter) / (max_perimeter - min_perimeter)
        color = interpolate_color(start_color, end_color, factor)
        drawing.add(drawing.polygon(points=clipped_poly, 
                            fill=svgwrite.rgb(*color, '%'), 
                            fill_opacity=0.7, 
                            stroke='blue', 
                            stroke_width=0.01,
                            stroke_opacity=0.5))

drawing.save()

# Continue with the rest of your code...

# def clip(value, min_value, max_value):
#     return max(min(value, max_value), min_value)

drawing = svgwrite.Drawing(filename2, size=('1000px', '1000px'), viewBox='0 0 10 10')


print(filename1)
print(filename2)