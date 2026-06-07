import cv2
import numpy as np

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


def density_level(density):
    if density < 5:
        return "Ít"
    if density < 15:
        return "Trung bình"
    return "Nhiều"
