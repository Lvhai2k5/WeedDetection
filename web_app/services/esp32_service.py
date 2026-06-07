import requests
import io
import cv2
import numpy as np
from PIL import Image

class ESP32Service:
    def __init__(self, ip="10.10.10.10"):
        self.ip = ip
        self.url_capture = f"http://{self.ip}/capture"
        self.url_spray = f"http://{self.ip}/spray"

    def set_ip(self, ip):
        self.ip = ip
        self.url_capture = f"http://{self.ip}/capture"
        self.url_spray = f"http://{self.ip}/spray"

    def check_connection(self):
        try:
            res = requests.get(self.url_capture, timeout=3)
            return res.status_code == 200
        except Exception as e:
            print(f"ESP32 Connection Error: {e}")
            return False

    def capture_image(self):
        try:
            res = requests.get(self.url_capture, timeout=3)
            if res.status_code == 200:
                img = Image.open(io.BytesIO(res.content)).convert("RGB")
                return np.array(img)
            return None
        except Exception as e:
            print(f"ESP32 Capture Error: {e}")
            return None

    def spray(self, spray_ms):
        try:
            url = f"{self.url_spray}?time={spray_ms}"
            requests.get(url, timeout=1)
            return True, f"💧 PHUN {spray_ms}/2000 ms"
        except requests.exceptions.Timeout:
            return True, f"💧 PHUN {spray_ms}/2000 ms (ESP32 đang xử lý)"
        except Exception as e:
            return False, f"❌ LỖI: Không gửi được lệnh phun! {e}"

esp32_service = ESP32Service()
