import sys
sys.path.insert(0, 'src')

import cv2
import numpy as np
from config import EXAMPLE_DIR
from preprocess import preprocess_image
from edge_detection import detect_edges
from texture_analysis import detect_texture_anomalies
from defect_detection import detect_edge_defects

img_path = 'data/examples/7.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Preprocess
preprocessed = preprocess_image(img)
binary = preprocessed[0] if isinstance(preprocessed, tuple) else preprocessed

# Edge detection
edges = detect_edges(binary)
print(f"Edges non-zero: {np.sum(edges > 0)}")

# Edge defects
edge_defects, edge_contours = detect_edge_defects(edges, binary)
print(f"Edge defects non-zero: {np.sum(edge_defects > 0)}")
print(f"Edge defect count: {len(edge_contours)}")

# Texture anomalies
texture_anomalies, texture_map = detect_texture_anomalies(binary, threshold_std=2.0)
print(f"Texture non-zero: {np.sum(texture_anomalies > 0)}")

# Analyze contours
contours_edge = cv2.findContours(edge_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
print(f"\nEdge defect contours: {len(contours_edge)}")
for i, cnt in enumerate(contours_edge):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"  Contour {i}: area={area:.1f}, bbox=({x},{y},{w},{h})")

contours_texture = cv2.findContours(texture_anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
print(f"\nTexture anomaly contours: {len(contours_texture)}")
for i, cnt in enumerate(contours_texture):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"  Contour {i}: area={area:.1f}, bbox=({x},{y},{w},{h})")
