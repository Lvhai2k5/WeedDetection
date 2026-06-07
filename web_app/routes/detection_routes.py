import cv2
import numpy as np
from flask import Blueprint, request, jsonify, Response
from services.detection_service import detection_service
from services.esp32_service import esp32_service
from models.database_manager import DatabaseManager

detection_bp = Blueprint('detection', __name__)
db_manager = DatabaseManager()

@detection_bp.route('/capture-detect', methods=['POST'])
def capture_detect():
    img_array = esp32_service.capture_image()
    if img_array is None:
        return jsonify({"success": False, "message": "Không thể chụp ảnh từ ESP32!"}), 500
        
    result = detection_service.process_image(img_array)
    
    # Save to history if successful and not blurred
    if result.get("success") and not result.get("blur_warning"):
        db_manager.save_detection({
            "image_path": result.get("saved_image_path", ""),
            "weed_density": result.get("density", 0.0),
            "young_density": result.get("young_density", 0.0),
            "mature_density": result.get("mature_density", 0.0),
            "weed_count": result.get("weed_count", 0),
            "spray_time": result.get("spray_ms", 0),
            "blur_score": result.get("blur_score", 0.0)
        })

    return jsonify(result)

@detection_bp.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Không tìm thấy file ảnh!"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Chưa chọn file!"}), 400
        
    # Read image from file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return jsonify({"success": False, "message": "Không đọc được ảnh (file hỏng hoặc sai định dạng)."}), 400
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = detection_service.process_image(img_rgb)
    
    # Save to history if successful and not blurred
    if result.get("success") and not result.get("blur_warning"):
        db_manager.save_detection({
            "image_path": result.get("saved_image_path", ""),
            "weed_density": result.get("density", 0.0),
            "young_density": result.get("young_density", 0.0),
            "mature_density": result.get("mature_density", 0.0),
            "weed_count": result.get("weed_count", 0),
            "spray_time": result.get("spray_ms", 0),
            "blur_score": result.get("blur_score", 0.0)
        })
        
    return jsonify(result)

def generate_frames():
    while True:
        img_array = esp32_service.capture_image()
        if img_array is not None:
            # We just stream the captured frame as-is, maybe resize it a bit for performance
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (540, 360))
            ret, buffer = cv2.imencode('.jpg', img_resized)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # ESP32 not reachable, just yield an empty frame or sleep
            import time
            time.sleep(1)

@detection_bp.route('/video-stream')
def video_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@detection_bp.route('/history', methods=['GET'])
def get_history():
    records = db_manager.get_history(limit=50)
    return jsonify({"success": True, "data": records})
