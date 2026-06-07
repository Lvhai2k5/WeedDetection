import cv2
import numpy as np
import base64
import os
from datetime import datetime

from utils.image_utils import (
    preprocess_image, check_blur, get_weed_mask_hsv, 
    classify_young_mature, calculate_density_percent, density_level
)
from patterns.factory.detector_factory import DetectorFactory
from patterns.strategy.spray_strategy import get_strategy_for_density, SprayContext

WEED_LABELS = {"Weed"}
BLUR_THRES = 80.0
CONF_THRES = 0.25
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DetectionService:
    def __init__(self):
        self.detector = DetectorFactory.get_detector("yolo")

    def process_image(self, img_rgb):
        logs = []
        result_data = {
            "success": False,
            "blur_warning": False,
            "weed_count": 0,
            "density": 0.0,
            "density_level": "None",
            "young_density": 0.0,
            "mature_density": 0.0,
            "spray_ms": 0,
            "images": {},
            "logs": logs,
            "saved_image_path": None
        }

        # 1. Preprocess
        img_rgb = preprocess_image(img_rgb)
        
        # 2. Blur check
        blur_value = check_blur(img_rgb)
        result_data["blur_score"] = blur_value
        
        if blur_value < BLUR_THRES:
            result_data["blur_warning"] = True
            logs.append(f"⚠ Ảnh quá mờ ({blur_value:.2f}) — không xử lý!")
            # Still return original image for display
            result_data["images"]["original"] = self._img_to_base64(img_rgb)
            return result_data

        # 3. YOLO Detect
        dets = self.detector.detect(img_rgb, conf_thres=CONF_THRES)
        weed_dets = [d for d in dets if d["label"] in WEED_LABELS]
        weed_count_yolo = len(weed_dets)
        result_data["weed_count"] = weed_count_yolo

        # 4. Mask HSV + Density
        weed_mask = get_weed_mask_hsv(img_rgb)
        density = calculate_density_percent(weed_mask)
        lvl = density_level(density)
        
        young_mask, mature_mask, young_color, mature_color = classify_young_mature(img_rgb, weed_mask)
        young_density = calculate_density_percent(young_mask)
        mature_density = calculate_density_percent(mature_mask)

        result_data["density"] = density
        result_data["density_level"] = lvl
        result_data["young_density"] = young_density
        result_data["mature_density"] = mature_density

        if weed_count_yolo == 0:
            logs.append("✅ YOLO: Không phát hiện CỎ (Weed) → DỪNG XỬ LÝ")
            result_data["images"]["yolo"] = self._img_to_base64(img_rgb)
            result_data["success"] = True
            return result_data
        
        logs.append("✅ YOLO detections:")
        for i, d in enumerate(dets, start=1):
            logs.append(f"[{i}] {d['label']} | conf={d['conf']:.2f} | area_bbox={d['area']}")

        logs.append(f"📊 Mật độ cỏ: {density:.2f}% → {lvl}")
        logs.append(f"- Cỏ non: {young_density:.2f}%")
        logs.append(f"- Cỏ trưởng thành: {mature_density:.2f}%")
        logs.append(f"- Độ nét ảnh: {blur_value:.2f}")

        # 5. Spray Decision using Strategy Pattern
        strategy = get_strategy_for_density(density)
        spray_context = SprayContext(strategy)
        spray_ms = spray_context.get_spray_time(density)
        result_data["spray_ms"] = spray_ms
        
        logs.append(f"💧 YOLO phát hiện {weed_count_yolo} chùm cỏ → CẦN PHUN {spray_ms}/2000 ms")

        # 6. Visualization
        spray_img = img_rgb.copy()
        # Overlay mask
        overlay = spray_img.copy()
        overlay[weed_mask > 0] = (overlay[weed_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        spray_img = overlay

        # Draw bbox
        for d in dets:
            x, y, w, h = d["bbox"]
            label = d["label"]
            conf = d["conf"]
            color = (255, 0, 0) if label in WEED_LABELS else (0, 255, 0)
            cv2.rectangle(spray_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                spray_img, f"{label} {conf:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

        # Generate base64 images for frontend
        mask_vis = cv2.cvtColor(weed_mask, cv2.COLOR_GRAY2RGB)
        
        result_data["images"]["yolo"] = self._img_to_base64(spray_img)
        result_data["images"]["mask"] = self._img_to_base64(mask_vis)
        result_data["images"]["young"] = self._img_to_base64(young_color)
        result_data["images"]["mature"] = self._img_to_base64(mature_color)
        
        # Save image for history
        filename = f"detect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(save_path, cv2.cvtColor(spray_img, cv2.COLOR_RGB2BGR))
        result_data["saved_image_path"] = f"uploads/{filename}"

        result_data["success"] = True
        return result_data

    def _img_to_base64(self, img_array):
        # Convert RGB to BGR for encoding
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

detection_service = DetectionService()
