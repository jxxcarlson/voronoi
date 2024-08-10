import numpy as np
from PIL import Image
import colorsys
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import svgwrite
import argparse
import os

def png_to_hsb_matrix(file_path):
    # Open the image file
    img = Image.open(file_path).convert('RGB')
    width, height = img.size

    # Initialize the HSB matrix as a NumPy array
    hsb_matrix = np.zeros((height, width, 3))

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hsb_matrix[y, x] = (h, s, v)

    return hsb_matrix

def chunk_hsb_matrix(hsb_matrix, chunk_size):
    height, width, _ = hsb_matrix.shape
    chunked_height = height // chunk_size
    chunked_width = width // chunk_size
    
    chunked_hsb_matrix = np.zeros((chunked_height, chunked_width, 3))
    
    for y in range(chunked_height):
        for x in range(chunked_width):
            chunk = hsb_matrix[y*chunk_size:(y+1)*chunk_size, 
                               x*chunk_size:(x+1)*chunk_size]
            chunked_hsb_matrix[y, x] = np.mean(chunk, axis=(0, 1))
    
    return chunked_hsb_matrix

def hsb_matrix_to_png(hsb_matrix, output_path, upscale_factor=1):
    height, width, _ = hsb_matrix.shape
    img = Image.new('RGB', (width * upscale_factor, height * upscale_factor))
    
    for y in range(height):
        for x in range(width):
            h, s, v = hsb_matrix[y, x]
            r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
            for i in range(upscale_factor):
                for j in range(upscale_factor):
                    img.putpixel((x * upscale_factor + i, y * upscale_factor + j), (r, g, b))
    
    img.save(output_path)
    print(f"Image saved to {output_path}")

def generate_random_points(hsb_matrix, N, size, f=lambda x: x):
    rows, columns, _ = hsb_matrix.shape
    aspect_ratio = columns / rows
    
    # Create the rectangle R
    width = size
    height = size / aspect_ratio
    
    # Initialize empty array for random points
    RP = []
    
    for _ in range(N):
        # Generate trial random point
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        # Find closest matching element in hsb_matrix
        row = int(y / height * rows)
        col = int(x / width * columns)
        
        # Get brightness value and calculate probability
        _, _, v = hsb_matrix[row, col]
        p = f(v)
        
        # Add point to RP with probability p
        if np.random.random() < p:
            RP.append([x, y])
    
    return np.array(RP)

def generate_random_points_with_color(hsb_matrix, N, size, f=lambda x: x):
    rows, columns, _ = hsb_matrix.shape
    aspect_ratio = columns / rows
    
    # Create the rectangle R
    width = size
    height = size / aspect_ratio
    
    # Initialize empty array for random points
    RP = []
    # Initialize empty array for random points with color data
    RP_Colors = []

    for _ in range(N):
        # Generate trial random point
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        # Find closest matching element in hsb_matrix
        row = int(y / height * rows)
        col = int(x / width * columns)
        
        # Get brightness value and calculate probability
        h, s, v = hsb_matrix[row, col]
        p = f(v)
        
        # Add point to RP with probability p
        if np.random.random() < p:
            RP.append([x, y])
            RP_Colors.append([x,y,h,s,v])
    
    return np.array(RP), np.array(RP_Colors)   

def voronoi_to_svg(points, size, aspect_ratio, output_path):
    print("size, size/aspect_ratio: ", size, size/aspect_ratio)
    
    # Filter points
    filtered_points, _ = remove_points_outside_bbox(points, (0, 0, size, size/aspect_ratio))
    
    # Compute Voronoi tessellation with filtered points
    vor = Voronoi(filtered_points)

    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, size=(f'{size}px', f'{size/aspect_ratio}px'))

    # Plot ridges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            start = vor.vertices[simplex[0]]
            end = vor.vertices[simplex[1]]
            dwg.add(dwg.line(start=start, end=end, stroke='black', stroke_width=0.5))

    # Save the SVG
    dwg.save()
    print(f"Voronoi SVG saved as {output_path}")

def plot_voronoi(points, size, aspect_ratio, output_path):
    # Compute Voronoi tessellation
    vor = Voronoi(points)
    print("Plotting voronoi, cells: ", len(vor.regions))

    print("Voronoi cells: ", len(vor.regions))
  
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot ridges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', linewidth=0.5)
    
    # Plot points
    # plt.scatter(points[:, 0], points[:, 1], s=1)
    print("Plotting points: ", len(points))
    
    # Set limits and aspect ratio
    plt.xlim(0, size)
    plt.ylim(0, size / aspect_ratio)
    ax.set_aspect('equal', adjustable='box')
    
    # Remove axes
    plt.axis('off')
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    print("Voronoi plot saved as " + output_path)
    
def get_voronoi_polygon_bounds(vor):
    """
    Get the smallest and largest coordinates of all polygons in a Voronoi diagram.
    
    :param vor: Voronoi tessellation object
    :return: tuple (min_x, min_y, max_x, max_y)
    """
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf
    
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = vor.vertices[region]
            if len(polygon):
                local_min_x, local_min_y = np.min(polygon, axis=0)
                local_max_x, local_max_y = np.max(polygon, axis=0)
                
                min_x = min(min_x, local_min_x)
                min_y = min(min_y, local_min_y)
                max_x = max(max_x, local_max_x)
                max_y = max(max_y, local_max_y)
    
    return min_x, min_y, max_x, max_y    
    # plt.show()

def clip_line_to_bbox(p1, p2, bbox):
    """Clip a line segment to a bounding box."""
    x1, y1 = p1
    x2, y2 = p2
    xmin, ymin, xmax, ymax = bbox
    
    def clip(p, q, edge):
        if edge == 0: return (xmin, p[1] + (q[1] - p[1]) * (xmin - p[0]) / (q[0] - p[0]))
        if edge == 1: return (xmax, p[1] + (q[1] - p[1]) * (xmax - p[0]) / (q[0] - p[0]))
        if edge == 2: return (p[0] + (q[0] - p[0]) * (ymin - p[1]) / (q[1] - p[1]), ymin)
        if edge == 3: return (p[0] + (q[0] - p[0]) * (ymax - p[1]) / (q[1] - p[1]), ymax)
    
    code1 = (int(x1 < xmin) | (int(x1 > xmax) << 1) | 
             (int(y1 < ymin) << 2) | (int(y1 > ymax) << 3))
    code2 = (int(x2 < xmin) | (int(x2 > xmax) << 1) | 
             (int(y2 < ymin) << 2) | (int(y2 > ymax) << 3))
    
    while code1 | code2:
        if code1 & code2:
            return None, None
        code = code1 if code1 else code2
        edge = 0
        if code & 8: edge = 3
        elif code & 4: edge = 2
        elif code & 2: edge = 1
        point = clip((x1, y1), (x2, y2), edge)
        if code == code1:
            x1, y1 = point
            code1 = (int(x1 < xmin) | (int(x1 > xmax) << 1) | 
                     (int(y1 < ymin) << 2) | (int(y1 > ymax) << 3))
        else:
            x2, y2 = point
            code2 = (int(x2 < xmin) | (int(x2 > xmax) << 1) | 
                     (int(y2 < ymin) << 2) | (int(y2 > ymax) << 3))
    
    return (x1, y1), (x2, y2)

def remove_points_outside_bbox(points, bbox):
    xmin, ymin, xmax, ymax = bbox
    
    # Identify points inside the bounding box
    mask = np.all((points >= [xmin, ymin]) & (points <= [xmax, ymax]), axis=1)
    points_inside = points[mask]
    kept_indices = np.where(mask)[0]
    
    return points_inside, kept_indices

def plot_voronoi_color2(points, colors, size, aspect_ratio, output_path):
    print("\n\nBefore filtering:")
    print(f"Number of points: {len(points)}")
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    print(f"Extreme points: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    filtered_points, kept_indices = remove_points_outside_bbox(points, (0, 0, size, size/aspect_ratio))
    filtered_colors = colors[kept_indices]
    
    print("\n\nAfter filtering:")
    print(f"Number of points: {len(filtered_points)}")
    min_x, min_y = np.min(filtered_points, axis=0)
    max_x, max_y = np.max(filtered_points, axis=0)
    print(f"Extreme points: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    # Compute Voronoi tessellation
    vor = Voronoi(filtered_points)
    
    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, size=(f'{size}px', f'{size/aspect_ratio}px'))

    # Create a background rectangle
    dwg.add(dwg.rect(insert=(0, 0), size=(f'{size}px', f'{size/aspect_ratio}px'), fill='white'))

    # Plot colored Voronoi cells
    print("Colored voronoi cells: ", len(vor.regions))
    for i, region in enumerate(vor.regions):
        if -1 not in region and len(region) > 2:  # Ensure the polygon is valid and has at least 3 points
            polygon = vor.vertices[region]
            if i < len(vor.point_region):
                point_index = vor.point_region[i]
                if point_index < len(filtered_colors):
                    color_data = filtered_colors[point_index]
                    h, s, v = color_data[-3:]  # Last three values are h, s, v
                    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
                    color = svgwrite.rgb(r, g, b)
                    dwg.add(dwg.polygon(points=polygon, fill=color, stroke='none'))

    # Plot Voronoi edges
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            start = vor.vertices[ridge[0]]
            end = vor.vertices[ridge[1]]
            dwg.add(dwg.line(start=start, end=end, stroke='black', stroke_width=0.5, stroke_opacity=0.5))

    # Save the SVG
    dwg.save()
    print(f"Colored Voronoi SVG saved as {output_path}")

def get_file_prefix(file_path):
    return os.path.splitext(file_path)[0]

def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

  

def main(file_path, n_voronoi_cells, chunk_size):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return
    

    file_name = get_file_name(file_path)
    image_out_prefix = 'images_out'
    # Create the 'images' directory if it doesn't exist
    os.makedirs(image_out_prefix, exist_ok=True)

    hsb_matrix = png_to_hsb_matrix(file_path)

    # Print the shape of the HSB matrix
    print(f"Shape of HSB matrix: {hsb_matrix.shape}")

    # Chunk the HSB matrix with a chunk size of 128x128
    # chunk_size = 32
    chunked_hsb_matrix = chunk_hsb_matrix(hsb_matrix, chunk_size)

    # Print the shape of the chunked HSB matrix
    print(f"Shape of chunked HSB matrix: {chunked_hsb_matrix.shape}")

    # Save the chunked HSB matrix as a PNG
    output_path = image_out_prefix + '/' + file_name + '_chunked.png'
    hsb_matrix_to_png(chunked_hsb_matrix, output_path, upscale_factor=chunk_size)

    # Generate random points
    # N = 32000  # Number of trial points
    N = n_voronoi_cells
    size = 1000  # Width of the rectangle R
    random_points, random_points_colors = generate_random_points_with_color(chunked_hsb_matrix, N, size, lambda x: x)

    # Calculate aspect ratio
    aspect_ratio = chunked_hsb_matrix.shape[1] / chunked_hsb_matrix.shape[0]

    # Create and plot Voronoi diagram
    print("Creating Voronoi diagram (plot)...")
    output_path = image_out_prefix + '/' + file_name + '_voronoi.png'
    plot_voronoi(random_points, size, aspect_ratio, output_path)

    # Generate SVG Voronoi diagram
    print("Creating Voronoi diagram (svg)...")
    output_path = image_out_prefix + '/' + file_name + '_voronoi.svg'
    voronoi_to_svg(random_points, size, aspect_ratio, output_path)

    # Create and save colored Voronoi diagram as SVG
    print("Creating colored Voronoi diagram (svg)...")
    output_path = image_out_prefix + '/' + file_name + '_voronoi_color.svg'
    plot_voronoi_color2(random_points, random_points_colors, size, aspect_ratio, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Voronoi diagrams from an image.")
    parser.add_argument("file_path", help="Path to the input image file")
    parser.add_argument("n_voronoi_cells", help="Number of voronoi cells")
    parser.add_argument("chunk_size", help="Size of chunks")


    args = parser.parse_args()

    # Usage:
    # python python/voronoi_from_image.py images_in/klee.png 16000 32
    main(args.file_path, int(args.n_voronoi_cells), int(args.chunk_size))