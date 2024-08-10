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
    # Compute Voronoi tessellation
    vor = Voronoi(points)

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
    
    # Display the plot
    # plt.show()


def plot_voronoi_color(points, colors, size, aspect_ratio, output_path):
    # Compute Voronoi tessellation
    vor = Voronoi(points)

    # Create SVG drawing
    dwg = svgwrite.Drawing(output_path, size=(f'{size}px', f'{size/aspect_ratio}px'))

    # Create a background rectangle
    dwg.add(dwg.rect(insert=(0, 0), size=(f'{size}px', f'{size/aspect_ratio}px'), fill='white'))

    # Create a dictionary to map points to colors
    point_to_color = {tuple(point[:2]): color[2:5] for point, color in zip(points, colors)}

    # Plot colored Voronoi cells
    print("Colored voronoi cells: ", len(vor.regions))
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = vor.vertices[region]
            if len(polygon) > 2:  # Ensure the polygon has at least 3 points
                # Find the point that corresponds to this region
                region_point = vor.points[vor.point_region == vor.regions.index(region)][0]
                h, s, v = point_to_color.get(tuple(region_point), (0, 0, 1))  # Default to white if not found
                r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
                color = svgwrite.rgb(r, g, b)
                dwg.add(dwg.polygon(points=polygon, fill=color, stroke='none'))

    # Plot Voronoi edges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            start = vor.vertices[simplex[0]]
            end = vor.vertices[simplex[1]]
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
    plot_voronoi_color(random_points, random_points_colors, size, aspect_ratio, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Voronoi diagrams from an image.")
    parser.add_argument("file_path", help="Path to the input image file")
    parser.add_argument("n_voronoi_cells", help="Number of voronoi cells")
    parser.add_argument("chunk_size", help="Size of chunks")


    args = parser.parse_args()

    main(args.file_path, int(args.n_voronoi_cells), int(args.chunk_size))