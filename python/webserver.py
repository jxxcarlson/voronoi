import os
from flask import Flask, render_template_string, request, send_from_directory
import shutil

app = Flask(__name__)

# Directory containing SVG files
SVG_DIR = '/Users/carlson/dev/generative/voronoi'
PORTFOLIO_DIR = os.path.join(SVG_DIR, 'portfolio')

# Ensure portfolio directory exists
if not os.path.exists(PORTFOLIO_DIR):
    os.makedirs(PORTFOLIO_DIR)

# HTML template
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>SVG Gallery</title>
    <style>
        body { background-color: black; color: white; font-family: Arial, sans-serif; }
        .image-container { display: flex; flex-direction: column; align-items: center; margin-bottom: 20px; }
        img { height: 80vh; max-width: 100%; object-fit: contain; }
        .buttons { margin-top: 10px; }
        button { margin: 0 10px; padding: 5px 10px; }
        .debug-info { color: yellow; }
        .error-message { color: red; }
    </style>

    <style>
    img {
        height: 80vh;
        max-width: 100%;
        object-fit: contain;
        background-color: white; /* Add this line */
    }
  <!DOCTYPE html>
<html>
<head>
    <title>SVG Gallery</title>
    <style>
        body { background-color: black; color: white; font-family: Arial, sans-serif; }
        .image-container { display: flex; flex-direction: column; align-items: center; margin-bottom: 20px; }
        img { height: 80vh; max-width: 100%; object-fit: contain; background-color: white; }
        .buttons { margin-top: 10px; }
        button { margin: 0 10px; padding: 5px 10px; }
    </style>
</head>
<body>
    {% for svg in svg_files %}
    <div class="image-container">
        <img src="{{ url_for('serve_svg', filename=svg) }}" alt="{{ svg }}">
        <div class="buttons">
            <button onclick="handleClick('keep', '{{ svg }}')">Keep</button>
            <button onclick="handleClick('remove', '{{ svg }}')">Remove</button>
        </div>
    </div>
    {% endfor %}

    <script>
    function handleClick(action, filename) {
        fetch('/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: 'action=' + action + '&filename=' + filename
        }).then(response => response.json())
        .then(data => {
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + data.message);
            }
        });
    }
    </script>
</body>
</html>
'''

@app.route('/svg/<path:filename>')
def serve_svg(filename):
    return send_from_directory(SVG_DIR, filename, mimetype='image/svg+xml')

@app.route('/svg/<path:filename>')
def serve_svg(filename):
    try:
        app.logger.info(f"Attempting to serve file: {filename}")
        full_path = os.path.join(SVG_DIR, filename)
        app.logger.info(f"Full path: {full_path}")
        if os.path.exists(full_path):
            app.logger.info(f"File exists: {full_path}")
            return send_from_directory(SVG_DIR, filename, mimetype='image/svg+xml')
        else:
            app.logger.error(f"File does not exist: {full_path}")
            abort(404)
    except Exception as e:
        app.logger.error(f"Error serving {filename}: {str(e)}")
        abort(404)


@app.route('/action', methods=['POST'])
def handle_action():
    action = request.form['action']
    filename = request.form['filename']
    file_path = os.path.join(SVG_DIR, filename)

    if action == 'keep':
        try:
            shutil.move(file_path, os.path.join(PORTFOLIO_DIR, filename))
            return {'success': True}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    elif action == 'remove':
        try:
            os.remove(file_path)
            return {'success': True}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    else:
        return {'success': False, 'message': 'Invalid action'}

if __name__ == '__main__':
    app.run(debug=True)                                                                                                                                                                                                                                                                                             