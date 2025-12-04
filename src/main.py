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
import csv
from datetime import datetime
from config import EXAMPLE_DIR, OUTPUT_DIR
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
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Tạo file CSV để lưu báo cáo
    report_path = os.path.join(OUTPUT_DIR, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    all_defects = []
    
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
            processed_img, _ = preprocess_image(gray)
            if processed_img is None:
                print(f"❌ Tiền xử lý thất bại {filename}, bỏ qua.")
                continue
            
            # 3) Edge detection
            print("→ Phát hiện biên...")
            edges = detect_edges(processed_img)
            if edges is None:
                print(f"❌ Edge detection thất bại {filename}, bỏ qua.")
                continue
            
            # 4) Phát hiện lỗi kết hợp (edge + texture)
            print("→ Phát hiện lỗi (kết hợp biên + texture)...")
            combined_mask, edge_defects, texture_defects, defect_info = detect_defects_combined(
                img, gray, edges
            )
            
            if len(defect_info) == 0:
                print(f"ℹ️  Không phát hiện lỗi trong ảnh {filename}")
                result_img = gray.copy()
            else:
                print(f"✓ Phát hiện {len(defect_info)} lỗi")
                
                # Vẽ kết quả
                result_img = visualize_defects_analysis(gray, combined_mask, edge_defects, texture_defects, defect_info)
                
                # In thông tin chi tiết
                for i, defect in enumerate(defect_info, 1):
                    print(f"  Lỗi {i}:")
                    print(f"    - Loại: {defect['type']}")
                    print(f"    - Diện tích: {defect['area']:.1f} px²")
                    print(f"    - Mức độ nghiêm trọng: {defect['severity']:.1f}%")
                    print(f"    - Texture entropy: {defect['texture_features']['entropy']:.3f}")
                    
                    # Lưu vào danh sách
                    all_defects.append({
                        'filename': filename,
                        'defect_id': i,
                        'type': defect['type'],
                        'area': f"{defect['area']:.1f}",
                        'width': defect['width'],
                        'height': defect['height'],
                        'severity': f"{defect['severity']:.1f}",
                        'entropy': f"{defect['texture_features']['entropy']:.3f}",
                        'contrast': f"{defect['texture_features']['contrast']:.3f}",
                        'position_x': defect['x'],
                        'position_y': defect['y']
                    })
            
            # 5) Lưu ảnh kết quả
            output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
            save_image(output_path, result_img)
            print(f"✓ Lưu kết quả: {output_path}")
    
    # 6) Lưu báo cáo CSV
    if all_defects:
        print(f"\n📊 Lưu báo cáo: {report_path}")
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_defects[0].keys())
            writer.writeheader()
            writer.writerows(all_defects)
        
        # In tóm tắt
        print(f"\n📈 TÓMLƯỢC:")
        print(f"  - Tổng số ảnh: {len(set(d['filename'] for d in all_defects))}")
        print(f"  - Tổng số lỗi: {len(all_defects)}")
        print(f"  - Phân loại lỗi:")
        for defect_type in set(d['type'] for d in all_defects):
            count = sum(1 for d in all_defects if d['type'] == defect_type)
            print(f"    • {defect_type}: {count}")
    else:
        print("\n✓ Xử lý hoàn tất - không phát hiện lỗi")

if __name__ == "__main__":
    process_all_examples()
