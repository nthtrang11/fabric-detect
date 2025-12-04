import cv2
import numpy as np
from config import EXAMPLE_DIR, PROCESSED_DIR
from preprocess import preprocess_image
from edge_detection import detect_edges
from texture_analysis import detect_texture_anomalies

img_path = r'data/examples/7.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

print('Original image shape:', img.shape)
print('Min/Max values:', img.min(), img.max())

# Preprocess
preprocessed = preprocess_image(img)
print('\nPreprocessed shape:', preprocessed.shape)
print('Preprocessed Min/Max:', preprocessed.min(), preprocessed.max())

# Edge detection
edges = detect_edges(preprocessed)
print('\nEdges shape:', edges.shape)
print('Edges non-zero pixels:', np.sum(edges > 0))

# Texture analysis
texture_anomalies = detect_texture_anomalies(preprocessed, threshold_std=2.0)
print('\nTexture anomalies shape:', texture_anomalies.shape)
print('Texture non-zero pixels:', np.sum(texture_anomalies > 0))

# Show stats
print('\n=== STATS ===')
print(f'Edge pixels: {np.sum(edges > 0)}')
print(f'Texture pixels: {np.sum(texture_anomalies > 0)}')
