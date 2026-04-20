import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression(img, D):
    """Applies Non-Maximum Suppression to thin the edges."""
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
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

def main():
    image_path = "images.jpeg"
    original_img = cv2.imread(image_path)
    
    if original_img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 1. Grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (Using cv2.GaussianBlur)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1.4)
    
    # 3. Sobel Edge Detection (Using cv2.Sobel)
    # Gradient in x direction
    Ix = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    # Gradient in y direction
    Iy = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and direction
    grad_magnitude = np.hypot(Ix, Iy)
    grad_magnitude = grad_magnitude / grad_magnitude.max() * 255
    grad_direction = np.arctan2(Iy, Ix)
    
    # 4. Non-Maximum Suppression
    nms_img = non_max_suppression(grad_magnitude, grad_direction)
    
    # 5. Double Thresholding
    threshold_img, weak, strong = threshold(nms_img, lowThresholdRatio=0.05, highThresholdRatio=0.15)
    
    # 6. Edge Tracking by Hysteresis
    final_img = hysteresis(threshold_img, weak, strong)

    # Visualization
    titles = ['Original Image', 'Grayscale', 'Gaussian Blur', 'Gradient Magnitude', 'Non-Max Suppression', 'Double Threshold', 'Final Edge MAP']
    images = [cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), gray_img, blurred_img, grad_magnitude, nms_img, threshold_img, final_img]

    plt.figure(figsize=(12, 12))
    for i in range(7):
        plt.subplot(3, 3, i+1)
        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('canny_steps_result.png')
    print("Results saved as 'canny_steps_result.png'")
    # plt.show() # Commented out to prevent blocking in headless environments

if __name__ == '__main__':
    main()
