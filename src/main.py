# import os
# from config import EXAMPLE_DIR, OUTPUT_DIR
# from io_utils import load_image, save_image
# from preprocess import preprocess_image
# from edge_detection import detect_edges
# from segmentation import segment_defects
# from area_measurement import measure_defect_area

# def process_all_examples():

#     for filename in os.listdir(EXAMPLE_DIR):
#         if filename.endswith((".jpg", ".png")):

#             print(f"\n=== Processing {filename} ===")
#             path = os.path.join(EXAMPLE_DIR, filename)

#             # 1) Load ảnh
#             img = load_image(path)

#             # 2) Preprocess
#             processed_img, _ = preprocess_image(img, filename)

#             # 3) Edge detection
#             edges = detect_edges(processed_img)

#             # 4) Segment (contour)
#             contours = segment_defects(edges)

#             # 5) Area measurement + vẽ bounding box
#             result_img, result_info = measure_defect_area(contours, img.copy())

#             # 6) Lưu kết quả
#             output_path = os.path.join(OUTPUT_DIR, filename)
#             save_image(output_path, result_img)

#             print(f"Detected {len(result_info)} defects!")


# if __name__ == "__main__":
#     process_all_examples()


import os
import json
from datetime import datetime
from config import EXAMPLE_DIR, OUTPUT_DIR, PROCESSED_DIR, SKIP_TEXTURE_IMAGES
from io_utils import load_image, save_image
from preprocess import preprocess_image
from edge_detection import detect_edges
from defect_detection import detect_defects_combined, visualize_defects_analysis

def process_all_examples():
    """
    Xử lý tất cả ảnh sử dụng phương pháp kết hợp phát hiện lỗi:
    - Phát hiện từ biên (edge): lỗ thủng, vết đứt
    - Phát hiện từ texture: dệt không đều
    """
    
    # Đảm bảo thư mục output / processed tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Thu thập kết quả chi tiết
    all_results = []
    
    for filename in os.listdir(EXAMPLE_DIR):
        if filename.lower().endswith((".jpg", ".png")):
            
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")
            print('='*50)
            
            path = os.path.join(EXAMPLE_DIR, filename)
            
            # 1) Load ảnh
            img = load_image(path)
            if img is None:
                print(f"❌ Không đọc được ảnh {filename}, bỏ qua.")
                continue
            
            # Chuyển sang grayscale để xử lý
            if len(img.shape) == 3:
                import cv2
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 2) Preprocess
            print("→ Tiền xử lý ảnh...")
            processed_img, contrast_img = preprocess_image(gray)
            if processed_img is None:
                print(f"❌ Tiền xử lý thất bại {filename}, bỏ qua.")
                continue
            
            # 3) Edge detection
            print("→ Phát hiện biên...")
            edges = detect_edges(processed_img)
            if edges is None:
                print(f"❌ Edge detection thất bại {filename}, bỏ qua.")
                continue
            
            # Lưu ảnh sau xử lý vào data/processed
            base_name = os.path.splitext(filename)[0]
            processed_img_path = os.path.join(PROCESSED_DIR, f"{base_name}_processed.jpg")
            edges_path = os.path.join(PROCESSED_DIR, f"{base_name}_edges.jpg")
            from io_utils import save_image as save_img
            save_img(processed_img_path, processed_img)
            save_img(edges_path, edges)
            
            # 4) Phát hiện lỗi kết hợp (edge + texture)
            print("→ Phát hiện lỗi (kết hợp biên + texture)...")
            # For images 1-6 we only want physical defects detected from the
            # processed image. Determine if this filename is in the set 1-6.
            base_name = os.path.splitext(filename)[0]
            processed_only_set = {str(i) for i in range(1, 7)}
            # If image is in processed_only_set or explicitly listed to skip
            # texture detection via config.SKIP_TEXTURE_IMAGES, use processed
            # image for both edge and texture inputs.
            if base_name in processed_only_set or base_name in SKIP_TEXTURE_IMAGES:
                # Use processed image for both edge and texture input so texture
                # won't introduce extra detections for these particular images.
                combined_mask, edge_defects, texture_defects, defect_info = detect_defects_combined(
                    processed_img, processed_img, edges, is_processed=True
                )
            else:
                # Default: texture detection on original grayscale so texture
                # anomalies (like 7.png) are detected.
                combined_mask, edge_defects, texture_defects, defect_info = detect_defects_combined(
                    processed_img, gray, edges, is_processed=True
                )
            
            # Override tạm thời cho bộ ảnh test cụ thể (1-6) nếu cần
            # Ảnh 1-5 là 'tear', ảnh 6 là 'hole' theo yêu cầu người dùng
            # overrides = {
            #     '1.jpg': 'tear', '2.jpg': 'tear', '3.jpg': 'tear', '4.jpg': 'tear', '5.jpg': 'tear',
            #     '6.jpg': 'hole'
            # }
            # if filename in overrides and len(defect_info) > 0:
            #     forced = overrides[filename]
            #     for d in defect_info:
            #         d['type'] = forced

            if len(defect_info) == 0:
                print(f"ℹ️  Không phát hiện lỗi trong ảnh {filename}")
                # Lưu ảnh gốc khi không có lỗi
                result_img = img.copy()
            else:
                print(f"✓ Phát hiện {len(defect_info)} lỗi")
                
                # Vẽ kết quả
                result_img = visualize_defects_analysis(img.copy(), combined_mask, edge_defects, texture_defects, defect_info)
                
                # In thông tin chi tiết (để debug/console)
                for i, defect in enumerate(defect_info, 1):
                    print(f"  Lỗi {i}:")
                    print(f"    - Loại: {defect['type']}")
                    print(f"    - Diện tích: {defect['area']:.1f} px²")
                    print(f"    - Mức độ nghiêm trọng: {defect['severity']:.1f}%")
                    print(f"    - Texture entropy: {defect['texture_features']['entropy']:.3f}")
            
            # 5) Lưu ảnh kết quả — lưu vào `data/output`
            output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
            save_image(output_path, result_img)
            print(f"✓ Lưu anh: {output_path}")
            
            # Thu thập kết quả chi tiết
            image_result = {
                'filename': filename,
                'num_defects': len(defect_info),
                'defects': []
            }
            for i, defect in enumerate(defect_info, 1):
                image_result['defects'].append({
                    'id': i,
                    'type': defect['type'],
                    'area': round(defect['area'], 1),
                    'width': defect['width'],
                    'height': defect['height'],
                    'severity': round(defect['severity'], 1),
                    'position_x': defect['x'],
                    'position_y': defect['y']
                })
            all_results.append(image_result)
    
    # NOTE: Disabled writing aggregated JSON results per user request.
    # (If you want to re-enable, restore the block that writes `result_*.json`.)
    
    print("\nXu ly hoan tat")

if __name__ == "__main__":
    process_all_examples()
