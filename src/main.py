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
    
    # Không tạo file báo cáo nữa — sẽ lưu ảnh gốc có chú thích lỗi
    
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
            
            # 5) Lưu ảnh kết quả — sử dụng ảnh gốc kèm chú thích
            output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
            save_image(output_path, result_img)
            print(f"✓ Lưu kết quả: {output_path}")
    
    print("\n✓ Xử lý hoàn tất")

if __name__ == "__main__":
    process_all_examples()
