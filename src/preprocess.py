# import cv2
# import numpy as np
# import os

# # Thư mục lưu ảnh đã xử lý
# PROCESSED_DIR = "data/processed"
# os.makedirs(PROCESSED_DIR, exist_ok=True)

# def preprocess_image_clean(image_input, filename=None,
#                            denoise_strength=25,
#                            median_ksize=7,
#                            gaussian_ksize=7,
#                            nlm_h=15,
#                            min_area=50,
#                            threshold_method="adaptive"):
#     """
#     Tiền xử lý ảnh nâng cao:
#     - Chuyển sang grayscale nếu ảnh màu
#     - Tăng tương phản với CLAHE
#     - Lọc nhiễu mạnh (median + bilateral + Gaussian + Non-local Means)
#     - Threshold sang trắng/đen (binary) với C=0
#     - Morphology loại bỏ các chấm sáng nhỏ
#     - Giữ lại các vùng sáng lớn (>= min_area)
#     - Lưu ảnh đã xử lý vào PROCESSED_DIR
#     """

#     # 1. Đọc ảnh
#     if isinstance(image_input, str):
#         img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
#     else:
#         if len(image_input.shape) == 3:
#             img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
#         else:
#             img = image_input.copy()

#     if img is None:
#         raise ValueError("Không đọc được ảnh!")

#     # 2. CLAHE để tăng tương phản
#     clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
#     contrast = clahe.apply(img)

#     # 3. Lọc nhiễu mạnh
#     denoised = cv2.medianBlur(contrast, median_ksize)
#     denoised = cv2.bilateralFilter(denoised, d=14,
#                                    sigmaColor=denoise_strength,
#                                    sigmaSpace=denoise_strength)
#     denoised = cv2.GaussianBlur(denoised, (gaussian_ksize, gaussian_ksize), 0)
#     denoised = cv2.fastNlMeansDenoising(denoised, h=nlm_h)

#     # 4. Threshold sang trắng/đen
#     if threshold_method == "otsu":
#         _, binary = cv2.threshold(denoised, 0, 255,
#                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         binary = cv2.adaptiveThreshold(
#             denoised, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             blockSize=31,
#             C=-1  # theo yêu cầu
#         )

#     # 5. Morphology loại bỏ chấm sáng nhỏ
#     kernel = np.ones((3,3), np.uint8)
#     binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

#     # 6. Giữ lại các vùng sáng lớn
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
#     binary_large = np.zeros_like(binary_clean)
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             binary_large[labels == i] = 255

#     # 7. Lưu ảnh đã xử lý
#     if filename is not None:
#         save_path = os.path.join(PROCESSED_DIR, filename)
#         cv2.imwrite(save_path, binary_large)

#     return binary_large, contrast

# # --- Giữ tên cũ để main.py không thay đổi ---
# preprocess_image = preprocess_image_clean

# # Ví dụ chạy thử
# if __name__ == "__main__":
#     img_path = "data/example.jpg"  # thay bằng ảnh của bạn
#     processed, contrast = preprocess_image(img_path, filename="example_processed.jpg")
#     cv2.imshow("Processed", processed)
#     cv2.imshow("Contrast", contrast)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import cv2
import numpy as np
import os

# Thư mục lưu ảnh đã xử lý
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_image_clean(image_input, filename=None,
                           denoise_strength=25,
                           median_ksize=7,
                           gaussian_ksize=7,
                           nlm_h=15,
                           min_area=50,
                           threshold_method="adaptive"):
    """
    Tiền xử lý ảnh nâng cao:
    - Chuyển sang grayscale nếu ảnh màu
    - Tăng tương phản với CLAHE
    - Lọc nhiễu mạnh (median + bilateral + Gaussian + Non-local Means)
    - Threshold sang trắng/đen (binary) với C=-1
    - Morphology loại bỏ các chấm sáng nhỏ
    - Giữ lại các vùng sáng lớn (>= min_area)
    - Lưu ảnh đã xử lý vào PROCESSED_DIR
    """

    # 1. Đọc ảnh
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    else:
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()

    if img is None:
        raise ValueError("Không đọc được ảnh!")

    # 2. CLAHE để tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    contrast = clahe.apply(img)

    # 3. Lọc nhiễu mạnh
    denoised = cv2.medianBlur(contrast, median_ksize)
    denoised = cv2.bilateralFilter(denoised, d=14,
                                   sigmaColor=denoise_strength,
                                   sigmaSpace=denoise_strength)
    denoised = cv2.GaussianBlur(denoised, (gaussian_ksize, gaussian_ksize), 0)
    denoised = cv2.fastNlMeansDenoising(denoised, h=nlm_h)

    # 4. Threshold sang trắng/đen
    if threshold_method == "otsu":
        _, binary = cv2.threshold(denoised, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=-1
        )

    # 5. Morphology loại bỏ chấm sáng nhỏ
    kernel = np.ones((3,3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 6. Giữ lại các vùng sáng lớn
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
    binary_large = np.zeros_like(binary_clean)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            binary_large[labels == i] = 255

    # 7. Lưu ảnh đã xử lý
    if filename is not None:
        save_path = os.path.join(PROCESSED_DIR, filename)
        cv2.imwrite(save_path, binary_large)

    return binary_large, contrast

# Giữ tên cũ
preprocess_image = preprocess_image_clean

# --- Test nhanh ---
if __name__ == "__main__":
    img_path = "data/example.jpg"
    processed, contrast = preprocess_image(img_path, filename="example_processed.jpg")
    cv2.imshow("Processed", processed)
    cv2.imshow("Contrast", contrast)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
