import os
import cv2
from abc import ABC, abstractmethod
from ultralytics import YOLO

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, img_rgb, conf_thres=0.25):
        pass


class YOLODetector(BaseDetector):
    def __init__(self, model_path):
        print(f"🔄 Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.classes = self.model.names
        print("✅ YOLO READY")

    def detect(self, img_rgb, conf_thres=0.25):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, conf=conf_thres, imgsz=640, device="cpu", verbose=False)[0]
        dets = []

        if results.boxes is None or len(results.boxes) == 0:
            return dets

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.classes[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            dets.append({
                "label": label,
                "conf": conf,
                "bbox": (x1, y1, w, h),
                "center": (cx, cy),
                "area": int(w * h)
            })

        return dets


class DetectorFactory:
    _instances = {}

    @staticmethod
    def get_detector(detector_type="yolo", model_path=None):
        if detector_type not in DetectorFactory._instances:
            if detector_type == "yolo":
                if model_path is None:
                    # Default path assuming app.py is run from web_app folder
                    model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model.pt')
                DetectorFactory._instances[detector_type] = YOLODetector(model_path)
            else:
                raise ValueError(f"Unknown detector type: {detector_type}")
        return DetectorFactory._instances[detector_type]
