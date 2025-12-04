import cv2
import numpy as np

def measure_defect_area(contours, img, defect_info=None):
    """
    Đo đạc diện tích các lỗi từ contours.
    
    Args:
        contours: danh sách contour từ cv2.findContours
        img: ảnh gốc để vẽ
        defect_info: thông tin defect (area, aspect_ratio, solidity) nếu có
    
    Returns:
        img: ảnh đã vẽ các bounding box
        results: danh sách dict với thông tin từng lỗi
    """
    results = []
    
    if not isinstance(contours, (list, tuple)) or len(contours) == 0:
        return img, results
    
    # Nếu contours là array, chuyển thành list
    if isinstance(contours, np.ndarray):
        contours = [contours]
    
    for idx, cnt in enumerate(contours):
        # Kiểm tra contour hợp lệ
        if cnt is None or not isinstance(cnt, np.ndarray) or len(cnt) < 3:
            continue
        
        try:
            area = cv2.contourArea(cnt)
            
            # Loại bỏ nhiễu nhỏ
            if area < 50:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Lấy thông tin từ defect_info nếu có
            aspect_ratio = 0
            solidity = 0
            if defect_info and idx < len(defect_info):
                aspect_ratio = defect_info[idx].get('aspect_ratio', 0)
                solidity = defect_info[idx].get('solidity', 0)
            
            # Vẽ bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Vẽ text thông tin
            text = f"Area: {area:.1f}px"
            cv2.putText(img, text, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            text2 = f"Ratio: {aspect_ratio:.2f}"
            cv2.putText(img, text2, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            results.append({
                "area": area,
                "bbox": (x, y, w, h),
                "width": w,
                "height": h,
                "aspect_ratio": aspect_ratio,
                "solidity": solidity
            })
        except Exception as e:
            print(f"Lỗi xử lý contour {idx}: {e}")
            continue
    
    return img, results
