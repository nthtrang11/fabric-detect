import cv2
import numpy as np
from texture_analysis import detect_texture_anomalies, extract_texture_descriptors

def detect_defects_combined(image, gray_image, edges_image, is_processed=True, physical_overlap_thresh=0.05):
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

        # Xác định loại lỗi (truyền contour để tính đặc trưng hình học)
        defect_type = classify_defect_type(gray_image, edge_defects, texture_defects, x, y, w, h, cnt, is_processed=is_processed)
        
        # Trích xuất texture features
        region = (y, x, h, w)
        texture_features = extract_texture_descriptors(gray_image, region)
        
        # Tính severity score (0-100)
        severity = calculate_defect_severity(area, w, h, texture_features, defect_type)
        
        # If defect classified as physical (hole/tear) we require that the
        # region has sufficient overlap with the processed mask `image`.
        if defect_type in ('hole', 'tear'):
            # `image` is expected to be the processed binary mask (0/255)
            try:
                proc_mask = (image > 0).astype('uint8')
                overlap = np.sum(proc_mask[y:y+h, x:x+w] > 0) / (w * h + 1e-10)
            except Exception:
                overlap = 0

            if overlap < physical_overlap_thresh:
                # Skip this physical detection because it doesn't appear in processed image
                continue

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

def classify_defect_type(gray_image, edge_mask, texture_mask, x, y, w, h, contour, is_processed=True):
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

    # Quyết định dựa trên edge vs texture:
    # - Nếu edge có ngưỡng đáng kể hoặc không quá nhỏ so với texture -> coi là edge-dominant
    # - Nếu texture rất lớn so với edge và edge rất nhỏ -> uneven_weaving
    # - Trường hợp mơ hồ: ưu tiên edge (giúp phát hiện scratches/tears)
    region_gray = gray_image[y:y+h, x:x+w]
    # Tính circularity trực tiếp từ contour
    circularity = compute_circularity_from_contour(contour)
    aspect_ratio = float(w) / (h + 1e-10)
    area_cnt = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = area_cnt / (hull_area + 1e-10)
    extent = area_cnt / (w * h + 1e-10)

    # Nếu edge có mặt rõ rệt (ưu tiên edge dù texture cũng có)
    # edge_dominant nếu edge đủ lớn hoặc chiếm tỉ lệ so với texture
    edge_dominant = (edge_ratio >= max(0.10, 0.25 * texture_ratio))
    if edge_dominant:
        # Nếu ảnh chưa được xử lý (raw) và người dùng muốn chỉ phát hiện physical
        # từ ảnh đã xử lý thì không classify thành 'hole'/'tear' ở đây.
        if not is_processed:
            # Tránh gán loại vật lý trên ảnh raw: gán 'combined' để biểu thị
            # vùng bất thường nhưng không xác định là physical defect từ raw.
            return 'combined'
        # Nếu vùng tròn, solidity cao và diện tích đủ lớn → hole
        if circularity > 0.60 and solidity > 0.55 and 0.6 <= aspect_ratio <= 1.6 and area_cnt > 180:
            return 'hole'
        # Nếu vùng kéo dài hoặc méo → tear (scratch)
        if aspect_ratio > 2.0 or aspect_ratio < 0.4 or circularity < 0.35 or extent < 0.25:
            return 'tear'
        # Các trường hợp khác: ưu tiên tear nếu shape không tròn
        if circularity < 0.50 or solidity < 0.5:
            return 'tear'
        return 'hole'

    # Nếu texture rõ rệt hơn -> uneven weaving
    # Nếu texture thực sự lớn còn edge rất nhỏ -> uneven weaving
    if texture_ratio > 0.60 and edge_ratio < 0.05:
        return 'uneven_weaving'

    # Fallback: nếu circularity lớn → hole, ngược lại coi là tear
    if circularity > 0.60 and solidity > 0.5:
        return 'hole'
    return 'tear'


def compute_circularity_from_contour(cnt):
    """
    Tính circularity từ contour: 4π * Area / Perimeter^2
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1e-10:
        return 0
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return min(max(circularity, 0.0), 1.0)

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

        # Vẽ text: tên lỗi + diện tích (px)
        text = f"{defect_type} | {defect['area']:.1f}px"

        # Compute text size and baseline
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Preferred position: above the bbox (x, y - 10). If that would put
        # the text outside the image, place it below the bbox instead.
        img_h, img_w = vis_img.shape[:2]
        text_x = x
        text_y_above = y - 10
        # top of text box would be text_y_above - text_h - baseline
        if text_y_above - text_h - baseline < 0:
            # place below bbox
            text_y = y + h + text_h + 10
            # ensure within image
            if text_y + baseline > img_h:
                text_y = img_h - baseline - 2
        else:
            text_y = text_y_above

        # Background rectangle for better contrast
        box_x1 = max(0, text_x)
        box_y1 = max(0, text_y - text_h - baseline)
        box_x2 = min(img_w, text_x + text_w)
        box_y2 = min(img_h, text_y + baseline)

        cv2.rectangle(vis_img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), cv2.FILLED)
        cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return vis_img
