import numpy as np
from PIL import Image
import colorsys
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

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

def generate_random_points(hsb_matrix, N, size):
    rows, columns, _ = hsb_matrix.shape
    aspect_ratio = columns / rows
    
    # Create the rectangle R
    aspect_ratio = columns / rows
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
        p = v
        
        # Add point to RP with probability p
        if np.random.random() < p:
            RP.append([x, y])
    
    return np.array(RP)


def plot_voronoi(points, size, aspect_ratio):
    # Compute Voronoi tessellation
    vor = Voronoi(points)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot ridges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', linewidth=0.5)
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], s=1)
    
    # Set limits and aspect ratio
    plt.xlim(0, size)
    plt.ylim(0, size / aspect_ratio)
    ax.set_aspect('equal', adjustable='box')
    
    # Remove axes
    plt.axis('off')
    
    # Save the plot
    plt.savefig('images/voronoi_plot.png', dpi=300, bbox_inches='tight', pad_inches=0)
    print("Voronoi plot saved as images/voronoi_plot.png")
    
    # Display the plot
    plt.show()

######################


# Example usage
file_path = 'images/klee.png'
hsb_matrix = png_to_hsb_matrix(file_path)

# Print the shape of the HSB matrix
print(f"Shape of HSB matrix: {hsb_matrix.shape}")

# Print the shape of the original HSB matrix
print(f"Shape of original HSB matrix: {hsb_matrix.shape}")

# Chunk the HSB matrix with a chunk size of 32x32
chunk_size = 128
chunked_hsb_matrix = chunk_hsb_matrix(hsb_matrix, chunk_size)

# Print the shape of the chunked HSB matrix
print(f"Shape of chunked HSB matrix: {chunked_hsb_matrix.shape}")

# Save the chunked HSB matrix as a PNG
output_path = 'images/chunked_klee.png'
hsb_matrix_to_png(chunked_hsb_matrix, output_path, upscale_factor=chunk_size)

# Generate random points
N = 1000  # Number of trial points
size = 1000  # Width of the rectangle R
random_points = generate_random_points(chunked_hsb_matrix, N, size)

# Display the points using matplotlib
# plt.figure(figsize=(10, 10))
# plt.scatter(random_points[:, 0], random_points[:, 1], s=1, alpha=0.5)
# plt.title(f"Random Points Based on HSB Matrix (N={N})")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# # Save the plot as an image
# plt.savefig('images/random_points_plot.png')
# print("Plot saved as images/random_points_plot.png")

# Create and plot Voronoi diagram
print("Creating Voronoi diagram...")
plot_voronoi(random_points, size, 1.2)