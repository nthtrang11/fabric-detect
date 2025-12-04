import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

def compute_lbp(image, radius=1, n_points=8):
    """
    Tính Local Binary Pattern (LBP) để phân tích texture cục bộ.
    LBP so sánh mỗi pixel với các pixel lân cận để tạo ra mẫu nhị phân.
    """
    height, width = image.shape
    lbp = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = image[y, x]
            binary_string = ""
            
            # So sánh với 8 hàng xóm
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                nx = x + radius * np.cos(angle)
                ny = y + radius * np.sin(angle)
                
                # Interpolation hai tuyến tính
                nx_floor = int(np.floor(nx))
                ny_floor = int(np.floor(ny))
                nx_ceil = nx_floor + 1
                ny_ceil = ny_floor + 1
                
                if 0 <= nx_ceil < width and 0 <= ny_ceil < height:
                    dx = nx - nx_floor
                    dy = ny - ny_floor
                    
                    I1 = image[ny_floor, nx_floor]
                    I2 = image[ny_floor, nx_ceil]
                    I3 = image[ny_ceil, nx_floor]
                    I4 = image[ny_ceil, nx_ceil]
                    
                    neighbor = (1-dx)*(1-dy)*I1 + dx*(1-dy)*I2 + (1-dx)*dy*I3 + dx*dy*I4
                    binary_string += "1" if neighbor >= center else "0"
            
            lbp[y, x] = int(binary_string, 2)
    
    return lbp

def compute_glcm_features(image, distances=[1], angles=[0], levels=256):
    """
    Tính GLCM (Gray Level Co-occurrence Matrix) features để phân tích texture.
    Các features: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    """
    from skimage.feature import greycomatrix, greycoprops
    
    # Normalize ảnh về 0-255
    image = ((image - image.min()) * (levels - 1) / (image.max() - image.min() + 1e-5)).astype(np.uint8)
    
    # Tính GLCM
    glcm = greycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    # Trích xuất features
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    asm = greycoprops(glcm, 'asm')[0, 0]
    
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'asm': asm
    }

def compute_local_texture_features(image, window_size=32, stride=16):
    """
    Tính toán texture features tại từng vùng cục bộ.
    Trả về một map của texture features trên toàn ảnh.
    """
    height, width = image.shape
    features_map = np.zeros((height, width), dtype=np.float32)
    
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            # Lấy vùng cục bộ
            patch = image[y:y+window_size, x:x+window_size]
            
            # Tính entropy texture (độ không đều của texture)
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Điền vào toàn bộ patch
            features_map[y:y+window_size, x:x+window_size] = entropy
    
    return features_map

def detect_texture_anomalies(image, window_size=32, threshold_std=1.5):
    """
    Phát hiện các vùng có texture bất thường so với phần còn lại.
    
    Args:
        image: ảnh grayscale đầu vào
        window_size: kích thước cửa sổ phân tích texture
        threshold_std: ngưỡng độ lệch chuẩn để coi là anomaly
    
    Returns:
        anomaly_mask: ảnh nhị phân với 255 là vùng bất thường
        texture_map: map của texture values
    """
    height, width = image.shape
    texture_map = np.zeros((height, width), dtype=np.float32)
    
    # Tính toán texture entropy cho từng vùng
    stride = window_size // 2
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            patch = image[y:y+window_size, x:x+window_size]
            
            # Tính độ không đều (entropy)
            hist, _ = np.histogram(patch, bins=256, range=(0, 256))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Tính độ lệch chuẩn (variation)
            variation = np.std(patch)
            
            # Combine entropy và variation
            texture_score = entropy * variation
            texture_map[y:y+window_size, x:x+window_size] = texture_score
    
    # Smooth texture map
    texture_map = cv2.GaussianBlur(texture_map, (5, 5), 0)
    
    # Tính giá trị trung bình và độ lệch chuẩn của texture
    mean_texture = np.mean(texture_map)
    std_texture = np.std(texture_map)
    
    # Ngưỡng: texture bất thường có giá trị > mean + threshold_std * std
    threshold = mean_texture + threshold_std * std_texture
    anomaly_mask = np.zeros((height, width), dtype=np.uint8)
    anomaly_mask[texture_map > threshold] = 255
    
    # Morphological operations để làm sạch
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
    
    return anomaly_mask, texture_map

def extract_texture_descriptors(image, region=None):
    """
    Trích xuất các descriptor texture từ ảnh hoặc một vùng cụ thể.
    
    Returns:
        dict: các texture features (mean, std, entropy, contrast, energy)
    """
    if region is not None:
        y, x, h, w = region
        patch = image[y:y+h, x:x+w]
    else:
        patch = image
    
    # Các thống kê cơ bản
    mean_val = np.mean(patch)
    std_val = np.std(patch)
    
    # Entropy
    hist, _ = np.histogram(patch, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Contrast (độ tương phản)
    gradient_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    contrast = np.mean(np.sqrt(gradient_x**2 + gradient_y**2))
    
    # Energy (năng lượng)
    energy = np.sum(patch.astype(np.float32)**2) / (patch.size + 1e-10)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'entropy': entropy,
        'contrast': contrast,
        'energy': energy
    }

def compare_texture_regions(image, region1, region2):
    """
    So sánh texture giữa hai vùng để xác định độ tương đồng.
    
    Returns:
        similarity_score: điểm tương đồng từ 0 (hoàn toàn khác) đến 1 (hoàn toàn giống)
    """
    features1 = extract_texture_descriptors(image, region1)
    features2 = extract_texture_descriptors(image, region2)
    
    # Tính khoảng cách Euclidean giữa các feature vectors
    features_list1 = np.array([features1[k] for k in ['mean', 'std', 'entropy', 'contrast', 'energy']])
    features_list2 = np.array([features2[k] for k in ['mean', 'std', 'entropy', 'contrast', 'energy']])
    
    # Normalize
    max_val = np.maximum(np.abs(features_list1), np.abs(features_list2))
    max_val[max_val == 0] = 1e-10
    features_list1 = features_list1 / max_val
    features_list2 = features_list2 / max_val
    
    # Tính khoảng cách Euclidean
    distance = np.sqrt(np.sum((features_list1 - features_list2)**2))
    
    # Chuyển đổi thành similarity score (0-1)
    similarity_score = np.exp(-distance)
    
    return similarity_score

def analyze_weaving_pattern(image, window_size=16):
    """
    Phân tích mẫu dệt bằng cách tính FFT để phát hiện các irregularities.
    Lỗi dệt không đều sẽ thay đổi spectrum của FFT.
    """
    height, width = image.shape
    pattern_anomaly = np.zeros((height, width), dtype=np.float32)
    
    stride = window_size // 2
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            patch = image[y:y+window_size, x:x+window_size].astype(np.float32)
            
            # FFT
            f_transform = np.fft.fft2(patch)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Tính entropy của magnitude spectrum
            magnitude_norm = magnitude / (magnitude.sum() + 1e-10)
            spectrum_entropy = -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10))
            
            # Điều đó có anomaly cao là entropy cao (dệt không đều)
            pattern_anomaly[y:y+window_size, x:x+window_size] = spectrum_entropy
    
    return pattern_anomaly
