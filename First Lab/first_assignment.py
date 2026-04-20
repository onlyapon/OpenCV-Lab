import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image1.jpeg')
img1 = cv2.imread('image2.jpeg')

images = np.concatenate((img, img1), axis=1)

# cv2.imshow("Original Images (Color)", images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if 'img' in locals() and 'img1' in locals():
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist1 = cv2.calcHist([gray_img1], [0], None, [256], [0, 256])

    # # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Image1 Histogram")
    plt.xlabel('bins')
    plt.ylabel("No of pixels")
    plt.plot(hist)

    # plt.subplot(122)
    # plt.title("Image2 Histogram")
    # plt.xlabel('bins')
    # plt.ylabel("No of pixels")
    # plt.plot(hist1)
    # plt.show()

# Apply Histogram Equalization
gray_img_eqhist = cv2.equalizeHist(gray_img)
gray_img1_eqhist = cv2.equalizeHist(gray_img1)

# Concatenate and display equalized images
eqhist_images = np.concatenate((gray_img_eqhist, gray_img1_eqhist), axis=1)
# cv2.imshow("Gray Scale Histogram Equalization", eqhist_images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Calculate and plot new histograms after equalization
hist_eq = cv2.calcHist([gray_img_eqhist], [0], None, [256], [0, 256])
hist1_eq = cv2.calcHist([gray_img1_eqhist], [0], None, [256], [0, 256])

# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.title("Image1 Equalized Histogram")
# plt.plot(hist_eq)

# plt.subplot(122)
# plt.title("Image2 Equalized Histogram")
# plt.plot(hist1_eq)
# plt.show()

# Create CLAHE object (clipLimit=40, tileGridSize is default 8x8)
clahe = cv2.createCLAHE(clipLimit=40)

# Apply CLAHE to the equalized images
gray_img_clahe = clahe.apply(gray_img_eqhist)
gray_img1_clahe = clahe.apply(gray_img1_eqhist)

# Concatenate and display CLAHE processed images
clahe_images = np.concatenate((gray_img_clahe, gray_img1_clahe), axis=1)
# cv2.imshow("CLAHE Processed Images", clahe_images)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

th = 80
max_val = 255

# Apply different thresholding types to the first CLAHE image (gray_img_clahe)
ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY)
ret, o2 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY_INV)
ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
ret, o4 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO_INV)
ret, o5 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)

# Note: OTSU thresholding (o6) typically ignores the 'th' parameter and calculates an optimal one.
ret_otsu, o6 = cv2.threshold(gray_img_clahe, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Add text for visualization (as done in the document)
cv2.putText(o1, "Thresh_Binary", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(o2, "Thresh_Binary_inv", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(o3, "Thresh_Tozero", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(o4, "Thresh_Tozero_inv", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(o5, "Thresh_trunc", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(o6, "Thresh_OSTU", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

# Concatenate and display results
final_thresh1 = np.concatenate((o1, o2, o3), axis=1)
final_thresh2 = np.concatenate((o4, o5, o6), axis=1)

# cv2.imshow("Thresholding Results 1", final_thresh1)
# cv2.imshow("Thresholding Results 2", final_thresh2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save images (as done in the document)
cv2.imwrite("Image1.jpg", final_thresh1)
cv2.imwrite("Image2.jpg", final_thresh2)


# Re-load grayscale images (as done in the document for adaptive thresholding)
gray_image = cv2.imread('cylinder1.png', 0)
gray_image1 = cv2.imread('cylinder.png', 0)

# Adaptive Thresholding for gray_image (Image 1)
# Different adaptive methods, block sizes, and C values are tested
thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4)

# Adaptive Thresholding for gray_image1 (Image 2)
thresh11 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh21 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
thresh31 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
thresh41 = cv2.adaptiveThreshold(gray_image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

# Concatenate and display
final_adaptive1 = np.concatenate((thresh1, thresh2, thresh3, thresh4), axis=1)
final_adaptive2 = np.concatenate((thresh11, thresh21, thresh31, thresh41), axis=1)

cv2.imshow("Adaptive Thresholding Image 1", final_adaptive1)
cv2.imshow("Adaptive Thresholding Image 2", final_adaptive2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save images
cv2.imwrite('rect.jpg', final_adaptive1)
cv2.imwrite('rect1.jpg', final_adaptive2)

# Otsu's Binarization for both images
ret_otsu1, thresh_otsu1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret_otsu2, thresh_otsu2 = cv2.threshold(gray_image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Concatenate and display final enhanced images
final_otsu = np.concatenate((thresh_otsu1, thresh_otsu2), axis=1)
cv2.imshow("Final Otsu's Binarization", final_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
cv2.imwrite('rect.jpeg', final_otsu)