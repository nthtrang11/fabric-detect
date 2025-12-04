import cv2
import numpy as np
from config import CANNY_THRESHOLD_1, CANNY_THRESHOLD_2

def detect_edges(processed_img):
    # Handle case where processed_img is a tuple (binary, contrast)
    if isinstance(processed_img, tuple):
        processed_img = processed_img[0]
    
    # Ensure it's a numpy array
    if not isinstance(processed_img, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(processed_img)}")
    
    edges = cv2.Canny(processed_img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    return edges
