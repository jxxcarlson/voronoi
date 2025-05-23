# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python project for generating artistic Voronoi decompositions from images. The project creates Voronoi diagrams with colored cells based on various algorithms and can process input images to create mosaic-like outputs.

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv myenv

# Activate environment
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Running Programs

#### Generate mosaic from image
```bash
python python/mosaic_from_image.py images_in/kandinsky-1908-1.png 256000 1
```
Format: `python python/mosaic_from_image.py <input-png-path> <number-of-points> <chunk-size>`

#### Generate standalone Voronoi diagrams
```bash
python python/voronoi.py
```

#### Run web server for SVG gallery
```bash
python python/webserver.py
```
Then navigate to http://localhost:5000 to view the gallery.

## Architecture

### Core Components

- **lib.py**: Core utility functions for Voronoi generation, polygon manipulation, color interpolation, and geometric operations
- **drawing.py**: SVG drawing functions for creating black & white and colored Voronoi diagrams with multiple color schemes
- **voronoi.py**: Main script for generating standalone Voronoi decompositions with various color schemes
- **mosaic_from_image.py**: Converts PNG images to Voronoi mosaics by sampling colors and generating points based on image data
- **webserver.py**: Flask-based web server for viewing generated SVG files in a gallery format with keep/remove functionality
- **hsb_values.py**: Utility for analyzing HSB color values in images

### Key Dependencies

- scipy.spatial.Voronoi: Core Voronoi computation
- svgwrite: SVG generation
- PIL/Pillow: Image processing
- matplotlib: Plotting support  
- numpy: Numerical operations
- Flask: Web server
- cairosvg: SVG to PNG conversion

### Output Files

- Generated files use timestamped names (e.g., `color_2024.01.15-14.30.45.svg`)
- SVG files are the primary output format
- PNG conversion is supported via cairosvg
- Output files are saved to the project root by default