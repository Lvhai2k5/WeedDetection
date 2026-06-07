from flask import Blueprint, request, jsonify
from services.esp32_service import esp32_service

esp32_bp = Blueprint('esp32', __name__)

@esp32_bp.route('/connect-esp32', methods=['POST'])
def connect_esp32():
    data = request.json
    ip = data.get('ip')
    if not ip:
        return jsonify({"success": False, "message": "No IP provided"}), 400
    
    esp32_service.set_ip(ip)
    is_connected = esp32_service.check_connection()
    
    if is_connected:
        return jsonify({"success": True, "message": f"Kết nối ESP32 thành công!\nIP: {ip}"})
    else:
        return jsonify({"success": False, "message": f"Không kết nối được ESP32 tại IP: {ip}"}), 500

@esp32_bp.route('/spray', methods=['POST'])
def spray():
    data = request.json
    spray_ms = data.get('time', 0)
    
    if spray_ms <= 0:
        return jsonify({"success": False, "message": "Thời gian phun không hợp lệ"}), 400
        
    success, message = esp32_service.spray(spray_ms)
    
    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "message": message}), 500
