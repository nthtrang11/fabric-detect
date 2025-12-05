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

# Texture filtering thresholds
# Minimum area (px) for an uneven_weaving region to be reported
MIN_TEXTURE_AREA = 700
# Minimum texture entropy for an uneven_weaving region to be reported
MIN_TEXTURE_ENTROPY = 1.0

# List of example base filenames (without extension) for which texture
# detection should be skipped/ignored (treat as processed-only for texture).
# Example: to skip texture for image `22.jpg` set SKIP_TEXTURE_IMAGES = {'22'}
SKIP_TEXTURE_IMAGES = set()

# Minimum area (px) for a hole to be reported
MIN_HOLE_AREA = 300
