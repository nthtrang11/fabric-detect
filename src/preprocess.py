import cv2
import os
from config import PROCESSED_DIR
from io_utils import save_image
import numpy as np

def _gamma_correction(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess_image(image, filename):
    """
    Tiền xử lý ảnh:
    - Grayscale
    - CLAHE cực mạnh
    - Contrast stretching cực đại
    - Khử nhiễu cực mạnh: bilateral + median + gaussian
    - Gamma correction để nền tối, defect sáng
    - Morphological top-hat nhấn defect
    """

    # 1) Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) CLAHE cực mạnh
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3) Contrast stretching theo percentiles 0.5% - 99.5%
    p_low, p_high = 0.5, 99.5
    v_min, v_max = np.percentile(enhanced, (p_low, p_high))
    stretched = np.clip((enhanced.astype(np.float32) - v_min) * 255.0 / (v_max - v_min), 0, 255).astype(np.uint8)

    # 4) Khử nhiễu cực mạnh
    denoise = cv2.bilateralFilter(stretched, d=11, sigmaColor=450, sigmaSpace=450)
    denoise = cv2.medianBlur(denoise, 15)  # loại bỏ mọi nhiễu hạt nhỏ
    denoise = cv2.GaussianBlur(denoise, (3, 3), 0)  # mượt nền, giữ defect

    # 5) Gamma correction
    gamma = 2.0
    gamma_corrected = _gamma_correction(denoise, gamma)

    # 6) Morphological top-hat
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gamma_corrected, cv2.MORPH_TOPHAT, kernel)

    # 7) Kết hợp gamma + top-hat
    combined = cv2.addWeighted(gamma_corrected, 1.0, tophat, 2.5, 0)
    final = np.clip(combined, 0, 255).astype(np.uint8)

    # 8) Lưu ảnh
    processed_path = os.path.join(PROCESSED_DIR, filename)
    save_image(processed_path, final)

    return final, processed_path
