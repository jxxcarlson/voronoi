from datetime import datetime
from scipy.spatial import Voronoi, cKDTree
import time
from collections import defaultdict
import numpy as np
import math
import random


# TIMESTAMPLE FILE NAMES

def get_timestamp():
    now = datetime.now()
    return now.strftime("%Y.%m.%d-%H.%M.%S")


def make_file_name(f):
    return f + '_' + get_timestamp() + '.svg'


# POLYGONS

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


def polygon_perimeter(poly):
    return sum(math.sqrt((x2-x1)**2 + (y2-y1)**2) for (x1, y1), (x2, y2) in zip(poly, poly[1:] + [poly[0]]))

def interpolate_color(color1, color2, factor):
    return tuple(int(color1[i] + (color2[i] - color1[i]) * factor) for i in range(3))

def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


# VORONOI CELLS

def get_voronoi_polygons(vor):
    polygons = []
    for region in vor.regions:
        if len(region) > 0 and all(i != -1 for i in region):
            polygon = [vor.vertices[i] for i in region]
            polygons.append(polygon)
    return polygons

# Probability of creating a point at (x,y)
def p(x,y):
    if x - y < -1 or x < 2:
        return 1.0
    if x - y < 2 or y < 2:
        return 0.5
    else:
        return 0.25

def voronoiCells(num_points):

    # num_points = 2560  # You can adjust this number
    points_ = np.random.rand(num_points, 2) * 10  # Create 2D array of points and scale the coordinates to [0, 10).
        
    # Create a mask based on the probability function:
    # np.array([p(x, y) for x, y in points_]) creates 
    # an array of probabilities by applying p(x,y) to
    # points_ .
    mask = np.random.random(len(points_)) < np.array([p(x, y) for x, y in points_])

    # Apply the mask to points_ to create points
    points = points_[mask]

    print("len(points_) = ", len(points_), "len(points) = ", len(points), "fraction = {:.3f}".format(len(points)/len(points_)))

    # Compute the Voronoi diagram using scipy.spatial
    return Voronoi(points)


# RANDOM WALK

def random_walk(voronoi_polygons, steps, timeout=30):
    start_time = time.time()
    neighbors = find_neighboring_polygons(voronoi_polygons)
    current = random.randint(0, len(voronoi_polygons) - 1)
    walk = [current]
    
    for _ in range(steps - 1):
        if time.time() - start_time > timeout:
            print(f"Random walk timed out after {timeout} seconds. Returning partial walk.")
            break
        if neighbors[current]:
            current = random.choice(list(neighbors[current]))
            walk.append(current)
        else:
            break  # Stop if we reach a polygon with no neighbors
    
    return walk

def find_neighboring_polygons(voronoi_polygons, tolerance=1e-6):
    neighbors = defaultdict(set)
    all_edges = []
    edge_centers = []
    for i, poly in enumerate(voronoi_polygons):
        for j in range(len(poly)):
            edge = (tuple(poly[j]), tuple(poly[(j+1) % len(poly)]))
            all_edges.append((i, edge))
            edge_centers.append(np.mean(edge, axis=0))
    
    edge_centers = np.array(edge_centers)
    if edge_centers.size == 0:
        print("Warning: No edge centers found. Check if voronoi_polygons is empty or incorrectly formatted.")
        return neighbors

    tree = cKDTree(edge_centers)
    
    for i, (poly_idx, edge) in enumerate(all_edges):
        edge_center = np.mean(edge, axis=0)
        potential_neighbors = tree.query_ball_point(edge_center, r=tolerance)
        
        for j in potential_neighbors:
            if i != j:
                other_poly_idx, other_edge = all_edges[j]
                if np.allclose(edge[0], other_edge[0], atol=tolerance) and np.allclose(edge[1], other_edge[1], atol=tolerance) or \
                   np.allclose(edge[0], other_edge[1], atol=tolerance) and np.allclose(edge[1], other_edge[0], atol=tolerance):
                    neighbors[poly_idx].add(other_poly_idx)
                    neighbors[other_poly_idx].add(poly_idx)
    
    return neighbors


# STATISTICS

# Function to format number to 4 significant figures


def statistics(data):

    def format_4sig(x):
        return f"{x:.4g}"

    sorted_data = sorted(data)

    # Print the min, max, and median of N
    print(f"Minimum perimeter: {format_4sig(sorted_data[0])}")
    print(f"Maximum perimeter: {format_4sig(sorted_data[-1])}")
    print(f"Median perimeter: {format_4sig(sorted_data[len(sorted_data)//2])}")

    # Print percentiles
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    print("Len percentiles:")
    for p in percentiles:
        print(f"{p}th percentile: {format_4sig(np.percentile(sorted_data, p))}")
