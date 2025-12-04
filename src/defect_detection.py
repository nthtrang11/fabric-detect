import cv2
import numpy as np
from texture_analysis import detect_texture_anomalies, extract_texture_descriptors

def detect_defects_combined(image, gray_image, edges_image):
    """
    Phát hiện lỗi kết hợp:
    1. Phát hiện lỗi từ biên (vết đứt, lỗ thủng)
    2. Phát hiện lỗi từ texture (dệt không đều)
    
    Returns:
        combined_mask: ảnh nhị phân kết hợp cả hai phương pháp
        edge_defects: lỗi phát hiện từ biên
        texture_defects: lỗi phát hiện từ texture
        defect_info: thông tin chi tiết về các lỗi
    """
    
    # Phương pháp 1: Phát hiện lỗi từ biên
    edge_defects, edge_info = detect_edge_defects(edges_image, gray_image)
    
    # Phương pháp 2: Phát hiện lỗi từ texture
    texture_defects, texture_map = detect_texture_anomalies(gray_image, window_size=32, threshold_std=1.5)
    
    # Kết hợp hai phương pháp
    combined_mask = cv2.bitwise_or(edge_defects, texture_defects)
    
    # Làm sạch
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Tìm contours từ combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Phân tích từng lỗi
    defect_info = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # Loại bỏ nhiễu nhỏ
            continue
        
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Xác định loại lỗi
        defect_type = classify_defect_type(gray_image, edge_defects, texture_defects, x, y, w, h)
        
        # Trích xuất texture features
        region = (y, x, h, w)
        texture_features = extract_texture_descriptors(gray_image, region)
        
        # Tính severity score (0-100)
        severity = calculate_defect_severity(area, w, h, texture_features, defect_type)
        
        defect_info.append({
            'area': area,
            'bbox': (x, y, w, h),
            'width': w,
            'height': h,
            'type': defect_type,
            'severity': severity,
            'texture_features': texture_features,
            'x': x,
            'y': y
        })
    
    return combined_mask, edge_defects, texture_defects, defect_info

def detect_edge_defects(edges, gray_image):
    """
    Phát hiện lỗi dựa trên các cạnh được phát hiện.
    Lỗi vết đứt, lỗ thủng sẽ được highlight.
    """
    # Dilate để nối các biên
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Tìm contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo mask từ các contours lớn
    edge_mask = np.zeros(edges.shape, dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 50000:  # Lọc kích thước hợp lý
            cv2.drawContours(edge_mask, [cnt], 0, 255, -1)
    
    return edge_mask, contours

def classify_defect_type(gray_image, edge_mask, texture_mask, x, y, w, h):
    """
    Phân loại loại lỗi dựa trên characteristics:
    - 'hole': Lỗ thủng (phát hiện từ edge)
    - 'tear': Vết đứt (phát hiện từ edge)
    - 'uneven_weaving': Dệt không đều (phát hiện từ texture)
    - 'combined': Kết hợp nhiều lỗi
    """
    
    # Lấy vùng lỗi
    region_edge = edge_mask[y:y+h, x:x+w]
    region_texture = texture_mask[y:y+h, x:x+w]
    
    # Tính tỷ lệ edge và texture trong vùng
    edge_ratio = np.sum(region_edge > 0) / (w * h + 1e-10)
    texture_ratio = np.sum(region_texture > 0) / (w * h + 1e-10)
    
    # Phân loại dựa trên tỷ lệ
    if edge_ratio > 0.6 and texture_ratio < 0.3:
        # Chủ yếu là edge → lỗ thủng hoặc vết đứt
        region_gray = gray_image[y:y+h, x:x+w]
        circularity = compute_circularity(region_edge, w, h)
        if circularity > 0.7:
            return 'hole'  # Lỗ thủng
        else:
            return 'tear'  # Vết đứt
    elif edge_ratio < 0.3 and texture_ratio > 0.6:
        return 'uneven_weaving'  # Dệt không đều
    else:
        return 'combined'  # Kết hợp

def compute_circularity(mask, width, height):
    """
    Tính độ tròn (circularity) của một vùng.
    Circularity = 4π * Area / Perimeter^2
    Giá trị gần 1 là vòng tròn, gần 0 là hình dài
    """
    area = np.sum(mask > 0)
    # Tính perimeter từ contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0
    
    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter < 1e-10:
        return 0
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return min(circularity, 1.0)  # Cap tại 1.0

def calculate_defect_severity(area, width, height, texture_features, defect_type):
    """
    Tính mức độ nghiêm trọng của lỗi (0-100).
    Dựa trên: diện tích, kích thước, texture variation, và loại lỗi.
    """
    severity = 0
    
    # 1. Dựa trên diện tích (0-40)
    area_severity = min(area / 1000.0 * 40, 40)
    
    # 2. Dựa trên kích thước (0-30)
    max_dim = max(width, height)
    size_severity = min(max_dim / 100.0 * 30, 30)
    
    # 3. Dựa trên texture (0-20)
    texture_severity = min(texture_features['entropy'] * 20, 20)
    
    # 4. Dựa trên loại lỗi (0-10)
    type_severity = {
        'hole': 10,           # Lỗ thủng nguy hiểm nhất
        'tear': 8,            # Vết đứt
        'uneven_weaving': 5,  # Dệt không đều ít nguy hiểm hơn
        'combined': 10        # Kết hợp
    }.get(defect_type, 0)
    
    severity = area_severity + size_severity + texture_severity + type_severity
    
    return min(severity, 100)  # Cap tại 100

def visualize_defects_analysis(image, combined_mask, edge_defects, texture_defects, defect_info):
    """
    Vẽ kết quả phân tích lỗi lên ảnh với các màu sắc khác nhau.
    - Đỏ: Lỗ thủng (hole)
    - Cam: Vết đứt (tear)
    - Xanh lục: Dệt không đều (uneven_weaving)
    - Tím: Kết hợp (combined)
    """
    vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    color_map = {
        'hole': (0, 0, 255),           # Đỏ
        'tear': (0, 165, 255),         # Cam
        'uneven_weaving': (0, 255, 0), # Xanh lục
        'combined': (255, 0, 255)      # Tím
    }
    
    for defect in defect_info:
        x, y, w, h = defect['bbox']
        defect_type = defect['type']
        severity = defect['severity']
        color = color_map.get(defect_type, (255, 255, 255))
        
        # Vẽ bounding box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
        
        # Vẽ text thông tin
        text = f"{defect_type}: Sev {severity:.0f}%"
        cv2.putText(vis_img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Vẽ area
        text_area = f"A:{defect['area']:.0f}"
        cv2.putText(vis_img, text_area, (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_img
