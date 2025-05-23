"""Flask web server for viewing and managing SVG Voronoi diagrams.

This module provides a web interface to:
- Browse generated SVG files in a gallery format
- Keep selected SVGs by moving them to a portfolio directory
- Remove unwanted SVGs
- View SVGs with a black background for better contrast

The server displays SVG files from a configured directory and allows
users to curate their collection through a simple web interface.
"""

import os
from flask import Flask, render_template_string, request, send_from_directory, jsonify
import shutil
from typing import Dict, List, Any
import logging

# Configure Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Directory configuration
SVG_DIR = '/Users/carlson/dev/generative/voronoi'
PORTFOLIO_DIR = os.path.join(SVG_DIR, 'portfolio')

# Ensure portfolio directory exists
os.makedirs(PORTFOLIO_DIR, exist_ok=True)

# HTML template for the gallery
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>SVG Voronoi Gallery</title>
    <style>
        body { 
            background-color: black; 
            color: white; 
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .gallery-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .image-container { 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            margin-bottom: 40px;
            border: 1px solid #333;
            padding: 20px;
            border-radius: 10px;
        }
        img { 
            height: 80vh; 
            max-width: 90%; 
            object-fit: contain; 
            background-color: white;
            border-radius: 5px;
        }
        .filename {
            margin-top: 10px;
            font-size: 0.9em;
            color: #aaa;
        }
        .buttons { 
            margin-top: 15px; 
        }
        button { 
            margin: 0 10px; 
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .keep-btn {
            background-color: #4CAF50;
            color: white;
        }
        .keep-btn:hover {
            background-color: #45a049;
        }
        .remove-btn {
            background-color: #f44336;
            color: white;
        }
        .remove-btn:hover {
            background-color: #da190b;
        }
        .status-message {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background-color: #4CAF50;
        }
        .error {
            background-color: #f44336;
        }
    </style>
</head>
<body>
    <div class="gallery-header">
        <h1>Voronoi SVG Gallery</h1>
        <p>{{ svg_count }} SVG files found</p>
    </div>
    
    <div id="statusMessage" class="status-message"></div>
    
    {% for svg in svg_files %}
    <div class="image-container" id="container-{{ loop.index }}">
        <img src="{{ url_for('serve_svg', filename=svg) }}" alt="{{ svg }}">
        <div class="filename">{{ svg }}</div>
        <div class="buttons">
            <button class="keep-btn" onclick="handleAction('keep', '{{ svg }}', {{ loop.index }})">
                Keep in Portfolio
            </button>
            <button class="remove-btn" onclick="handleAction('remove', '{{ svg }}', {{ loop.index }})">
                Remove
            </button>
        </div>
    </div>
    {% endfor %}

    <script>
    function showStatus(message, isSuccess) {
        const statusEl = document.getElementById('statusMessage');
        statusEl.textContent = message;
        statusEl.className = 'status-message ' + (isSuccess ? 'success' : 'error');
        statusEl.style.display = 'block';
        
        setTimeout(() => {
            statusEl.style.display = 'none';
        }, 3000);
    }
    
    function handleAction(action, filename, index) {
        fetch('/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'action=' + action + '&filename=' + encodeURIComponent(filename)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Hide the container
                const container = document.getElementById('container-' + index);
                container.style.display = 'none';
                
                // Show success message
                const actionText = action === 'keep' ? 'moved to portfolio' : 'removed';
                showStatus(`File ${actionText} successfully!`, true);
            } else {
                showStatus('Error: ' + data.message, false);
            }
        })
        .catch(error => {
            showStatus('Network error: ' + error, false);
        });
    }
    </script>
</body>
</html>
'''


@app.route('/')
def index() -> str:
    """Display the main gallery page with all SVG files."""
    try:
        # Get all SVG files in the directory
        svg_files = [f for f in os.listdir(SVG_DIR) 
                    if f.endswith('.svg') and os.path.isfile(os.path.join(SVG_DIR, f))]
        svg_files.sort(reverse=True)  # Most recent first
        
        return render_template_string(HTML_TEMPLATE, 
                                    svg_files=svg_files,
                                    svg_count=len(svg_files))
    except Exception as e:
        app.logger.error(f"Error loading gallery: {str(e)}")
        return f"Error loading gallery: {str(e)}", 500


@app.route('/svg/<path:filename>')
def serve_svg(filename: str):
    """Serve an SVG file from the configured directory.
    
    Args:
        filename: Name of the SVG file to serve
        
    Returns:
        SVG file content with appropriate MIME type
    """
    try:
        app.logger.info(f"Serving SVG: {filename}")
        return send_from_directory(SVG_DIR, filename, mimetype='image/svg+xml')
    except Exception as e:
        app.logger.error(f"Error serving {filename}: {str(e)}")
        return f"File not found: {filename}", 404


@app.route('/action', methods=['POST'])
def handle_action() -> Dict[str, Any]:
    """Handle keep/remove actions for SVG files.
    
    Returns:
        JSON response indicating success or failure
    """
    action = request.form.get('action')
    filename = request.form.get('filename')
    
    if not action or not filename:
        return jsonify({'success': False, 'message': 'Missing action or filename'})
    
    file_path = os.path.join(SVG_DIR, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': 'File not found'})
    
    try:
        if action == 'keep':
            # Move file to portfolio directory
            dest_path = os.path.join(PORTFOLIO_DIR, filename)
            shutil.move(file_path, dest_path)
            app.logger.info(f"Moved {filename} to portfolio")
            return jsonify({'success': True})
            
        elif action == 'remove':
            # Delete the file
            os.remove(file_path)
            app.logger.info(f"Removed {filename}")
            return jsonify({'success': True})
            
        else:
            return jsonify({'success': False, 'message': 'Invalid action'})
            
    except Exception as e:
        app.logger.error(f"Error handling {action} for {filename}: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/portfolio')
def portfolio() -> str:
    """Display SVG files in the portfolio directory."""
    try:
        svg_files = [f for f in os.listdir(PORTFOLIO_DIR) 
                    if f.endswith('.svg') and os.path.isfile(os.path.join(PORTFOLIO_DIR, f))]
        svg_files.sort(reverse=True)
        
        # Modify the template to show portfolio files
        portfolio_html = HTML_TEMPLATE.replace(
            '<h1>Voronoi SVG Gallery</h1>',
            '<h1>Portfolio Collection</h1>'
        ).replace(
            "url_for('serve_svg', filename=svg)",
            "url_for('serve_portfolio_svg', filename=svg)"
        )
        
        return render_template_string(portfolio_html, 
                                    svg_files=svg_files,
                                    svg_count=len(svg_files))
    except Exception as e:
        app.logger.error(f"Error loading portfolio: {str(e)}")
        return f"Error loading portfolio: {str(e)}", 500


@app.route('/portfolio/<path:filename>')
def serve_portfolio_svg(filename: str):
    """Serve an SVG file from the portfolio directory.
    
    Args:
        filename: Name of the SVG file to serve
        
    Returns:
        SVG file content with appropriate MIME type
    """
    try:
        return send_from_directory(PORTFOLIO_DIR, filename, mimetype='image/svg+xml')
    except Exception as e:
        app.logger.error(f"Error serving portfolio file {filename}: {str(e)}")
        return f"File not found: {filename}", 404


if __name__ == '__main__':
    print(f"Starting Voronoi Gallery Server")
    print(f"SVG Directory: {SVG_DIR}")
    print(f"Portfolio Directory: {PORTFOLIO_DIR}")
    print(f"Server running at: http://localhost:5000")
    print(f"Portfolio view at: http://localhost:5000/portfolio")
    
    app.run(debug=True, host='0.0.0.0', port=5000)