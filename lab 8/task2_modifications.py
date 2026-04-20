import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reuse functions from task1_canny.py or redefine them slightly modified

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    """Applies Double Thresholding."""
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    """Applies Hysteresis to track edges."""
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def non_max_suppression_standard(img, D):
    """Standard NMS with 0, 45, 90, 135 sectors."""
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
                # Standard sectors:
                # 0 +/- 22.5
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # 45 +/- 22.5
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # 90 +/- 22.5
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # 135 +/- 22.5
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    return Z

def non_max_suppression_modified(img, D):
    """Modified NMS by shifting angle bands.
    Let's shift the quantization by 22.5 degrees to see the effect.
    Usually we center around 0, 45, 90, 135.
    Let's center around 22.5, 67.5, 112.5, 157.5.
    This basically rotates the 'cone' of sensitivity.
    """
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
                # Modified sectors logic to force a change
                # Instead of standard bins, let's just perturb the logic significantly
                # or swap x and y roles, or change the ranges.
                
                # Simpler modification: Just clamp to 4 directions but with different boundaries
                # e.g., 0-45 -> Horizontal (!), 45-90 -> Diagonal 1, etc.
                
                if (0 <= angle[i,j] < 45):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (45 <= angle[i,j] < 90):
                     q = img[i+1, j-1]
                     r = img[i-1, j+1]
                elif (90 <= angle[i,j] < 135):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (135 <= angle[i,j] <= 180):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    return Z

def main():
    image_path = "images.jpeg"
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1.4)
    
    Ix = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_magnitude = np.hypot(Ix, Iy)
    grad_magnitude = grad_magnitude / grad_magnitude.max() * 255
    grad_direction = np.arctan2(Iy, Ix)
    
    # Task 2a: Modified Gradient Angle Directions
    # Standard NMS
    nms_standard = non_max_suppression_standard(grad_magnitude, grad_direction)
    # Modified NMS
    nms_modified = non_max_suppression_modified(grad_magnitude, grad_direction)
    
    # Process both to see final edge difference (using standard thresholds)
    res_std, w_std, s_std = threshold(nms_standard)
    final_std = hysteresis(res_std, w_std, s_std)
    
    res_mod, w_mod, s_mod = threshold(nms_modified)
    final_mod = hysteresis(res_mod, w_mod, s_mod)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(final_std, cmap='gray'), plt.title('Standard NMS Angles')
    plt.subplot(1, 2, 2), plt.imshow(final_mod, cmap='gray'), plt.title('Modified NMS Angles')
    plt.savefig('task2a_modified_angles.png')
    
    # Task 2b: Skip Non-Maximum Suppression
    # Feed raw Gradient Magnitude into Thresholding
    res_no_nms, w_no_nms, s_no_nms = threshold(grad_magnitude, lowThresholdRatio=0.05, highThresholdRatio=0.09)
    final_no_nms = hysteresis(res_no_nms, w_no_nms, s_no_nms)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(final_std, cmap='gray'), plt.title('With NMS')
    plt.subplot(1, 2, 2), plt.imshow(final_no_nms, cmap='gray'), plt.title('Without NMS')
    plt.savefig('task2b_no_nms.png')
    
    # Task 2c: Tuned Hysteresis Thresholds
    # Try finding "best" values. Let's try high threshold 0.2 vs standard 0.09
    res_tuned_1, w1, s1 = threshold(nms_standard, lowThresholdRatio=0.05, highThresholdRatio=0.20)
    final_tuned_1 = hysteresis(res_tuned_1, w1, s1)
    
    res_tuned_2, w2, s2 = threshold(nms_standard, lowThresholdRatio=0.10, highThresholdRatio=0.20)
    final_tuned_2 = hysteresis(res_tuned_2, w2, s2)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(final_std, cmap='gray'), plt.title('Default (L=0.05, H=0.09)')
    plt.subplot(1, 3, 2), plt.imshow(final_tuned_1, cmap='gray'), plt.title('Higher High (L=0.05, H=0.20)')
    plt.subplot(1, 3, 3), plt.imshow(final_tuned_2, cmap='gray'), plt.title('Strict (L=0.10, H=0.20)')
    plt.savefig('task2c_tuned.png')

if __name__ == '__main__':
    main()
