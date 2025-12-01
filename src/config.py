import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

EXAMPLE_DIR = os.path.join(DATA_DIR, "examples")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tham số tiền xử lý và phát hiện biên
BLUR_KERNEL = (5, 5)
CANNY_THRESHOLD_1 = 80
CANNY_THRESHOLD_2 = 150
