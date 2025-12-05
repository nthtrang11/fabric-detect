import cv2
import numpy as np

def segment_defects(binary_img):
    """
    Phân đoạn các cụm đốm/lỗi từ ảnh nhị phân đã preprocess.
    Trả về:
    - contours: danh sách contour tìm được
    - defect_info: danh sách dict với diện tích, aspect ratio, solidity của từng contour
    """

    if binary_img is None:
        raise ValueError("Ảnh nhị phân đầu vào None!")

    # Đảm bảo ảnh nhị phân 0/255
    binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)[1]

    # 1. Morphology: dilate để nối các điểm gần nhau
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(binary_img, kernel, iterations=2)

    # 2. Tìm contour
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 3. Phân tích contour
    defect_info = []
    for cnt in contours:
        if cnt is None or len(cnt) == 0:
            continue  # bỏ contour rỗng
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0

        defect_info.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        })

    return contours, defect_info

# --- Test nhanh ---
if __name__ == "__main__":
    img_path = "data/processed/example_processed.jpg"
    binary = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")

    contours, info = segment_defects(binary)

    # Vẽ bounding box lên ảnh gốc
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for d in info:
        cv2.rectangle(output, (d['x'], d['y']), (d['x']+d['w'], d['y']+d['h']), (0,0,255), 2)

    cv2.imshow("Defects", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # In thông tin các vùng
    print("Thông tin các vùng bất thường:", info)
