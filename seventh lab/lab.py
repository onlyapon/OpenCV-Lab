
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
img_path = '/Volumes/Blankspace/DIP and Robot Vision Lab/seventh lab/2.png'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image from {img_path}")
    exit()

# 2. Split into Input (Left) and Target (Right)
h, w, c = img.shape
midpoint = w // 2
input_noisy = img[:, :midpoint]
target_clean = img[:, midpoint:]

# 3. Apply Median Filter to remove Salt and Pepper Noise
# Analysis showed kernel size 5x5 gave the lowest MSE
denoised_img = cv2.medianBlur(input_noisy, 5)

# 4. Save the result
cv2.imwrite('result_denoised.png', denoised_img)
print("Denoised image saved as 'result_denoised.png'")

# Optional: Visualize Side-by-Side Comparison
# Concatenate: [Original Noisy | Denoised | Original Target]
comparison = np.hstack((input_noisy, denoised_img, target_clean))
cv2.imwrite('comparison_full.png', comparison)
print("Comparison saved as 'comparison_full.png'")

# Answer to questions
print("\n--- Answers ---")
print("1. What kind of noise is induced? -> Salt and Pepper Noise (Impulse Noise)")
print("2. How can we achieve the output image? -> By applying a Median Filter.")
