from flask import Flask
import os

# Fix for PyTorch 2.6+ unpickling error with older Ultralytics versions
import torch
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

from routes.esp32_routes import esp32_bp
from routes.detection_routes import detection_bp
from routes.ui_routes import ui_bp
from models.database_manager import DatabaseManager

app = Flask(__name__)

# Config
app.config['SECRET_KEY'] = 'weed_detection_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Database
db_manager = DatabaseManager()

# Register Blueprints
app.register_blueprint(ui_bp)
app.register_blueprint(esp32_bp, url_prefix='/api/esp32')
app.register_blueprint(detection_bp, url_prefix='/api/detection')

if __name__ == '__main__':
    # Initialize YOLO Model in background to avoid blocking server start?
    # It is loaded via Factory on first request, but we could pre-warm it here.
    from patterns.factory.detector_factory import DetectorFactory
    DetectorFactory.get_detector("yolo")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
