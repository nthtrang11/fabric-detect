import os
from config import EXAMPLE_DIR, OUTPUT_DIR
from io_utils import load_image, save_image
from preprocess import preprocess_image
from edge_detection import detect_edges
from segmentation import segment_defects
from area_measurement import measure_defect_area

def process_all_examples():

    for filename in os.listdir(EXAMPLE_DIR):
        if filename.endswith((".jpg", ".png")):

            print(f"\n=== Processing {filename} ===")
            path = os.path.join(EXAMPLE_DIR, filename)

            # 1) Load ảnh
            img = load_image(path)

            # 2) Preprocess
            processed_img, _ = preprocess_image(img, filename)

            # 3) Edge detection
            edges = detect_edges(processed_img)

            # 4) Segment (contour)
            contours = segment_defects(edges)

            # 5) Area measurement + vẽ bounding box
            result_img, result_info = measure_defect_area(contours, img.copy())

            # 6) Lưu kết quả
            output_path = os.path.join(OUTPUT_DIR, filename)
            save_image(output_path, result_img)

            print(f"Detected {len(result_info)} defects!")


if __name__ == "__main__":
    process_all_examples()
