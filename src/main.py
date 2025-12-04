import os
import cv2
from config import EXAMPLE_DIR, OUTPUT_DIR, PROCESSED_DIR
from io_utils import load_image, save_image
from preprocess import preprocess_image
from edge_detection import detect_edges
from defect_detection import detect_defects_combined, visualize_defects_analysis

def visualize_edge_defects(image, defect_info):
    """
    Chi ve cac loi phhat hien tu edge detection
    """
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()
    
    h_img, w_img = vis_img.shape[:2]
    
    color_map = {
        'hole': (0, 0, 255),           # Do
        'tear': (0, 165, 255),         # Cam
        'uneven_weaving': (0, 255, 0), # Xanh lac
        'combined': (255, 0, 255)      # Tim
    }
    
    for defect in defect_info:
        # Chi ve cac loi tu edge (hole, tear)
        if defect['type'] not in ['hole', 'tear']:
            continue
            
        x, y, w, h = defect['bbox']
        defect_type = defect['type']
        area = defect['area']
        color = color_map.get(defect_type, (255, 255, 255))
        
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 3)
        
        text_main = f"{defect_type.upper()}"
        text_area = f"Area: {area:.1f} px2"
        
        y_top = max(5, y - 35)
        x_left = max(5, x)
        cv2.rectangle(vis_img, (x_left - 2, y_top), (min(x_left + 180, w_img - 5), y_top + 30), (0, 0, 0), -1)
        cv2.putText(vis_img, text_main, (x_left + 5, y_top + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_bottom = min(h_img - 35, y + h + 5)
        cv2.rectangle(vis_img, (x_left - 2, y_bottom), (min(x_left + 180, w_img - 5), y_bottom + 30), (0, 0, 0), -1)
        cv2.putText(vis_img, text_area, (x_left + 5, y_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_img

def process_all_examples():
    """
    Xu ly tat ca anh su dung phuong phap ket hop phat hien loi:
    - Phat hien tu bien (edge): lo thung, vet dut
    - Phat hien tu texture: det khong deu
    - Luu anh da xu ly vao data/processed (chi edge defects)
    - Luu anh output (anh goc voi ca edge + texture defects) vao data/output
    """
    
    # Dam bao thu muc ton tai
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    for filename in os.listdir(EXAMPLE_DIR):
        if filename.lower().endswith((".jpg", ".png")):
            
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")
            print('='*50)
            
            path = os.path.join(EXAMPLE_DIR, filename)
            
            # 1) Load anh goc
            img = load_image(path)
            if img is None:
                print(f"[ERROR] Khong doc duoc anh {filename}, bo qua.")
                continue
            
            # Chuyen sang grayscale de xu ly
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 2) Preprocess
            print("[*] Tien xu ly anh...")
            processed_img, _ = preprocess_image(gray)
            if processed_img is None:
                print(f"[ERROR] Tien xu ly that bai {filename}, bo qua.")
                continue
            
            # 3) Edge detection
            print("[*] Phat hien bien...")
            edges = detect_edges(processed_img)
            if edges is None:
                print(f"[ERROR] Edge detection that bai {filename}, bo qua.")
                continue
            
            # 4) Phat hien loi ket hop (edge + texture)
            print("[*] Phat hien loi (ket hop bien + texture)...")
            combined_mask, edge_defects, texture_defects, defect_info = detect_defects_combined(
                img, gray, edges
            )
            
            # 5) Luu anh processed - chi co edge defects (hole, tear)
            edge_only_defects = [d for d in defect_info if d['type'] in ['hole', 'tear']]
            if len(edge_only_defects) > 0:
                processed_vis = visualize_edge_defects(processed_img, edge_only_defects)
            else:
                processed_vis = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR) if len(processed_img.shape) == 2 else processed_img.copy()
            
            processed_path = os.path.join(PROCESSED_DIR, f"processed_{filename}")
            save_image(processed_path, processed_vis)
            print(f"[OK] Luu anh xu ly: {processed_path} ({len(edge_only_defects)} edge defects)")
            
            # 6) Luu anh output - ca edge + texture defects
            if len(defect_info) == 0:
                print(f"[INFO] Khong phat hien loi trong anh {filename}")
                if len(img.shape) == 3:
                    result_img = img.copy()
                else:
                    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                print(f"[OK] Phat hien {len(defect_info)} loi (ket hop)")
                
                if len(img.shape) == 3:
                    result_img = img.copy()
                else:
                    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                result_img = visualize_defects_analysis(result_img, combined_mask, edge_defects, texture_defects, defect_info)
                
                for i, defect in enumerate(defect_info, 1):
                    print(f"  Loi {i}: {defect['type']} - Dien tich: {defect['area']:.1f} px2")
            
            output_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")
            save_image(output_path, result_img)
            print(f"[OK] Luu anh output: {output_path}")

if __name__ == "__main__":
    process_all_examples()
