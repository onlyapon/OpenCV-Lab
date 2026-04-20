
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image / 255.0 + gauss
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)

# 1. Load the clean image (Right half of 2.png)
img_path = '2.png'
full_img = cv2.imread(img_path)

if full_img is None:
    print("Error: Image not found.")
    exit()

h, w, c = full_img.shape
# The right half is the clean reference
clean_img = full_img[:, w//2:]

# 2. Add Gaussian Noise (since Bilateral works best on Gaussian/Random noise, not Salt & Pepper)
noisy_img = add_gaussian_noise(clean_img)

# 3. Apply Edge-Preserving Filters

# Filter A: Bilateral Filter
# Arguments: src, d (diameter), sigmaColor, sigmaSpace
# sigmaColor: standard deviation in the color space (how different colors must be to NOT mix)
# sigmaSpace: standard deviation in the coordinate space (how far pixels can be to influence each other)
bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# Filter B: Non-Local Means Denoising (NLM)
# A more advanced technique that looks for similar patches across the image
nlm = cv2.fastNlMeansDenoisingColored(noisy_img, None, 10, 10, 7, 21)

# Comparison: Gaussian Blur (Standard, non-edge-preserving)
gaussian_blur = cv2.GaussianBlur(noisy_img, (9, 9), 2)

# 4. Save and Compare
cv2.imwrite('task3_noise.png', noisy_img)
cv2.imwrite('task3_bilateral.png', bilateral)
cv2.imwrite('task3_nlm.png', nlm)
cv2.imwrite('task3_gaussian_blur.png', gaussian_blur)

print("Images saved: task3_noise.png, task3_bilateral.png, task3_nlm.png, task3_gaussian_blur.png")

# Justification Output
print("\n--- Justification ---")
print("1. Bilateral Filter: It smoothes images while preserving edges, by means of a nonlinear combination of nearby image values.")
print("   It weights pixels based on both spatial distance and intensity difference. Large intensity differences (edges) result in low weights, thus preserving the edge.")
print("2. Non-Local Means (NLM): It replaces a pixel with an average of all pixels in the image that have a similar neighborhood.")
print("   This is very effective at preserving textures and edges while reducing noise.")
