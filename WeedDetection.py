import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import requests
import io
import threading
import cv2
import numpy as np
import os

from ultralytics import YOLO
DEFAULT_ESP32_IP = "10.10.10.10"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model.pt")


CONF_THRES = 0.25
IMG_SIZE_SHOW = (540, 360)
WEED_LABELS = {"Weed"} 
BLUR_THRES = 80.0

YOLO_MODEL = None
YOLO_CLASSES = None

def load_yolo():
    global YOLO_MODEL, YOLO_CLASSES
    if YOLO_MODEL is None:
        print("🔄 Loading YOLO model...")
        YOLO_MODEL = YOLO(YOLO_PATH)
        YOLO_CLASSES = YOLO_MODEL.names
        print("✅ YOLO READY")


def preprocess_image(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def check_blur(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def yolo_detect(img_rgb, conf_thres=0.25):
    if YOLO_MODEL is None:
        return [] 
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = YOLO_MODEL(img_bgr,conf=conf_thres,imgsz=640,device="cpu",verbose=False)[0]
    dets = []

    if results.boxes is None or len(results.boxes) == 0:
        return dets

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = YOLO_CLASSES[cls_id]
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

def get_weed_mask_hsv(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    low = np.array([20, 20, 20])
    high = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    return mask


def classify_young_mature(img_rgb, weed_mask):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    young_mask = ((weed_mask > 0) & (S > 70) & (V > 120)).astype("uint8") * 255
    mature_mask = ((weed_mask > 0) & (S <= 70) & (V <= 120)).astype("uint8") * 255

    young_color = cv2.bitwise_and(img_rgb, img_rgb, mask=young_mask)
    mature_color = cv2.bitwise_and(img_rgb, img_rgb, mask=mature_mask)
    return young_mask, mature_mask, young_color, mature_color


def calculate_density_percent(mask_255):
    h, w = mask_255.shape[:2]
    weed_pixels = int(np.count_nonzero(mask_255))
    total_pixels = int(h * w)
    return (weed_pixels / total_pixels) * 100.0

def calculate_spray_time_ms(density_percent, max_ms=2000):
    if density_percent <= 0:
        return 0
    spray_ms = int(300 + (density_percent / 20.0) * (max_ms - 300))
    spray_ms = max(300, min(spray_ms, max_ms))
    return spray_ms


def density_level(density):
    if density < 5:
        return "Ít"
    if density < 15:
        return "Trung bình"
    return "Nhiều"

class WeedCamApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Ứng dụng nhận dạng cỏ, đánh giá mật độ và phun thuốc tự động")
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")


        # Trạng thái ESP32
        self.esp_ip = DEFAULT_ESP32_IP
        self.esp_url_capture = f"http://{self.esp_ip}/capture"
        self.esp_url_spray = f"http://{self.esp_ip}/spray"
        self.connected = False
        self.running = False

        # ================== LEFT ==================
        self.left_frame = ctk.CTkFrame(self, width=270, corner_radius=15)
        self.left_frame.pack(side="left", fill="y", padx=15, pady=15)

        ctk.CTkLabel(self.left_frame, text="🌿 KHUNG ĐIỀU KHIỂN HỆ THỐNG", font=("Arial", 20, "bold")).pack(pady=15)

        ctk.CTkLabel(self.left_frame, text="Nhập IP ESP32:", font=("Arial", 14)).pack()
        self.ip_entry = ctk.CTkEntry(self.left_frame, width=210)
        self.ip_entry.pack(pady=5)
        self.ip_entry.insert(0, DEFAULT_ESP32_IP)

        self.btn_connect = ctk.CTkButton(self.left_frame, text="🔌 Kết nối ESP32", command=self.connect_camera)
        self.btn_connect.pack(pady=5)

        self.btn_capture_process = ctk.CTkButton(
            self.left_frame, text="📷 Chụp & Nhận dạng", fg_color="#1f6aa5", command=self.capture_and_process
        )
        self.btn_capture_process.pack(pady=10)

        self.btn_start_stream = ctk.CTkButton(
            self.left_frame, text="▶ Xem trực tiếp", fg_color="#1f6aa5", command=self.start_stream
        )
        self.btn_start_stream.pack(pady=5)

        self.btn_stop_stream = ctk.CTkButton(
            self.left_frame, text="⏹ Dừng", fg_color="#8a1f1f", command=self.stop_stream
        )
        self.btn_stop_stream.pack(pady=5)

        ctk.CTkLabel(self.left_frame, text="────────────", font=("Arial", 14)).pack(pady=10)

        self.btn_open_test = ctk.CTkButton(self.left_frame, text="📁 Mở ảnh từ máy tính", command=self.open_test_image)
        self.btn_open_test.pack(pady=10)

        # Log
        self.result_text = ctk.CTkTextbox(self.left_frame, width=250, height=430, font=("Arial", 13))
        self.result_text.pack(pady=10)

        # ================== RIGHT ==================
        self.right_frame = ctk.CTkFrame(self, corner_radius=15)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=15, pady=15)

        titles = ["Mặt nạ cỏ", "Cỏ non", "Cỏ trưởng thành", "YOLO dự đoán"]
        self.img_labels = []

        for i in range(4):
            lbl_t = ctk.CTkLabel(self.right_frame, text=titles[i], font=("Arial", 18, "bold"))
            lbl_t.grid(row=(i // 2) * 2, column=i % 2, pady=5)

            lbl = ctk.CTkLabel(self.right_frame, text="")
            lbl.grid(row=(i // 2) * 2 + 1, column=i % 2, padx=10, pady=10)
            self.img_labels.append(lbl)

        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)
        threading.Thread(target=load_yolo, daemon=True).start()

    # ================= ESP32 ACTIONS =================

    def spray(self, spray_ms):
        try:
            url = f"{self.esp_url_spray}?time={spray_ms}"
            requests.get(url, timeout=1)
            self.result_text.insert("end", f"💧 PHUN {spray_ms}/2000 ms\n")

        except requests.exceptions.Timeout:
            self.result_text.insert(
                "end", f"💧 PHUN {spray_ms}/2000 ms (ESP32 đang xử lý)\n"
            )

        except Exception:
            self.result_text.insert("end", "❌ LỖI: Không gửi được lệnh phun!\n")


    def connect_camera(self):
        ip = self.ip_entry.get().strip()
        if ip == "":
            messagebox.showerror("Lỗi", "Vui lòng nhập IP ESP32!")
            return

        self.esp_ip = ip
        self.esp_url_capture = f"http://{ip}/capture"
        self.esp_url_spray = f"http://{ip}/spray"

        try:
            requests.get(self.esp_url_capture, timeout=3)
            self.connected = True
            messagebox.showinfo("OK", f"Kết nối ESP32 thành công!\nIP: {ip}")
        except Exception as e:
            self.connected = False
            messagebox.showerror("Lỗi", f"Không kết nối được ESP32!\n{e}")

    def capture_and_process(self):
        if not self.connected:
            messagebox.showwarning("Cảnh báo", "Chưa kết nối ESP32!")
            return

        try:
            res = requests.get(self.esp_url_capture, timeout=3)
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            self.process_and_display(np.array(img))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chụp ảnh!\n{e}")

    # ================= PC IMAGE =================
    def open_test_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh (file hỏng hoặc sai định dạng).")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.process_and_display(img_rgb)

    # ================= MAIN PIPELINE =================
    def process_and_display(self, img_rgb):
        self.result_text.delete("1.0", "end")

        # 1) preprocess
        img_rgb = preprocess_image(img_rgb)

        # 2) blur check
        blur_value = check_blur(img_rgb)
        if blur_value < BLUR_THRES:
            self.show_blur_warning(img_rgb, blur_value)
            return

        # 3) YOLO detect
        dets = yolo_detect(img_rgb, conf_thres=CONF_THRES)
        weed_dets = [d for d in dets if d["label"] in WEED_LABELS]
        weed_count_yolo = len(weed_dets)

        # 4) MASK HSV + density + young/mature
        weed_mask = get_weed_mask_hsv(img_rgb)
        density = calculate_density_percent(weed_mask)
        lvl = density_level(density)

        young_mask, mature_mask, young_color, mature_color = classify_young_mature(img_rgb, weed_mask)
        young_density = calculate_density_percent(young_mask)
        mature_density = calculate_density_percent(mature_mask)

        if weed_count_yolo == 0:
            self.result_text.insert(
                "end",
                "✅ YOLO: Không phát hiện CỎ (Weed) → DỪNG XỬ LÝ\n"
            )

            # Hiển thị ảnh gốc ở ô YOLO
            show = Image.fromarray(img_rgb.astype("uint8"))
            ct_img = ctk.CTkImage(show, size=IMG_SIZE_SHOW)
            self.img_labels[3].configure(image=ct_img, text="")
            self.img_labels[3].image = ct_img

            return   # ⛔ DỪNG TOÀN BỘ PIPELINE TẠI ĐÂY

        else:
            self.result_text.insert("end", "✅ YOLO detections:\n")
            for i, d in enumerate(dets, start=1):
                self.result_text.insert(
                    "end",
                    f"[{i}] {d['label']} | conf={d['conf']:.2f} | area_bbox={d['area']}\n"
                )

        # Log density + young/mature
        self.result_text.insert("end", "📊 Mật độ cỏ\n")
        self.result_text.insert("end", f"- Weed density: {density:.2f}%  → {lvl}\n")
        self.result_text.insert("end", f"- Cỏ non density: {young_density:.2f}%\n")
        self.result_text.insert("end", f"- Cỏ trưởng thành density: {mature_density:.2f}%\n")
        self.result_text.insert("end", f"- Độ nét ảnh: {blur_value:.2f}\n\n")

        # ================= MODIFIED: SPRAY DECISION =================
        if weed_count_yolo > 0:
            spray_ms = calculate_spray_time_ms(density)

            self.result_text.insert(
                "end",
                f"💧 YOLO phát hiện {weed_count_yolo} chùm cỏ → PHUN {spray_ms}/2000 ms\n"
            )

            self.spray(spray_ms)
        else:
            self.result_text.insert("end", "✅ YOLO: Không có cỏ\n")


        # 5) vẽ YOLO + overlay vùng mask (đẹp cho demo)
        spray_img = img_rgb.copy()

        # Overlay mask cỏ (nhẹ nhàng)
        overlay = spray_img.copy()
        overlay[weed_mask > 0] = (overlay[weed_mask > 0] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        spray_img = overlay

        # Vẽ bbox YOLO
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

        # 6) hiển thị 4 ô
        mask_vis = cv2.cvtColor(weed_mask, cv2.COLOR_GRAY2RGB)  # để show cho đẹp
        imgs = [mask_vis, young_color, mature_color, spray_img]

        for i, im in enumerate(imgs):
            show = Image.fromarray(im.astype("uint8"))
            ct_img = ctk.CTkImage(show, size=IMG_SIZE_SHOW)
            self.img_labels[i].configure(image=ct_img, text="")
            self.img_labels[i].image = ct_img

    # ================= BLUR WARNING =================
    def show_blur_warning(self, img_rgb, blur_value):
        blank = np.zeros_like(img_rgb)
        imgs = [blank, blank, blank, img_rgb]

        for i, im in enumerate(imgs):
            show = Image.fromarray(im.astype("uint8"))
            ct_img = ctk.CTkImage(show, size=IMG_SIZE_SHOW)
            self.img_labels[i].configure(image=ct_img, text="")
            self.img_labels[i].image = ct_img

        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"⚠ Ảnh quá mờ ({blur_value:.2f}) — không xử lý!\n")

    # ================= STREAM =================
    def start_stream(self):
        if not self.connected:
            messagebox.showwarning("Cảnh báo", "Chưa kết nối ESP32!")
            return
        if self.running:
            return
        self.running = True
        threading.Thread(target=self.update_stream, daemon=True).start()

    def update_stream(self):
        while self.running:
            try:
                res = requests.get(self.esp_url_capture, timeout=1)
                img = Image.open(io.BytesIO(res.content)).convert("RGB")
                ct_img = ctk.CTkImage(img, size=IMG_SIZE_SHOW)
                # hiển thị tạm ở ô cuối (YOLO bbox + vùng cỏ)
                self.img_labels[3].configure(image=ct_img, text="")
                self.img_labels[3].image = ct_img
            except:
                pass

    def stop_stream(self):
        self.running = False
        self.img_labels[3].configure(text="Video đã dừng", image=None)
        self.img_labels[3].image = None
 
if __name__ == "__main__":
    app = WeedCamApp()
    app.mainloop()
