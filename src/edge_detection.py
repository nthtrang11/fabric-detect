import cv2
from config import CANNY_THRESHOLD_1, CANNY_THRESHOLD_2

def detect_edges(processed_img):
    edges = cv2.Canny(processed_img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    return edges
