
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image1.jpeg')
img1 = cv2.imread('image2.jpeg')


# Re-load images in grayscale mode for a clean start
gray_image = cv2.imread('image1.jpeg', 0) 
gray_image1 = cv2.imread('image2.jpeg', 0) 

if gray_image is None or gray_image1 is None:
    print("Error: Could not load grayscale images. Check file paths.")
else:
    # Otsu's Binarization for Image 1: The threshold value is ignored (set to 0) 
    # when THRESH_OTSU is used, as the optimal value is calculated automatically.
    ret_otsu1, thresh_otsu1 = cv2.threshold(gray_image, 0, 255, 
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    # Otsu's Binarization for Image 2
    ret_otsu2, thresh_otsu2 = cv2.threshold(gray_image1, 0, 255, 
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    # Concatenate and display final enhanced images
    final_otsu = np.concatenate((thresh_otsu1, thresh_otsu2), axis=1)
    cv2.imshow("Otsu's Binarization Final Result", final_otsu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('rect.jpeg', final_otsu) 