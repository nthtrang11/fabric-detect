import cv2

def measure_defect_area(contours, img):
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 50:
            continue    # Loại nhiễu nhỏ

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img, f"Area: {area:.1f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        results.append({"area": area, "bbox": (x,y,w,h)})
    return img, results
