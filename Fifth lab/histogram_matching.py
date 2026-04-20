
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def create_bimodal_gaussian_histogram(m1, s1, m2, s2, weight1=0.5, weight2=0.5):
    """Creates a bimodal gaussian distribution to be used as the target histogram."""
    x = np.arange(256)
    g1 = np.exp(-((x - m1)**2) / (2 * s1**2))
    g2 = np.exp(-((x - m2)**2) / (2 * s2**2))
    target_hist = weight1 * g1 + weight2 * g2
    # Normalize to create a probability distribution
    if target_hist.sum() > 0:
        target_hist /= target_hist.sum()
    return target_hist

def histogram_matching(src_img, target_hist):
    """Matches the histogram of a source image to a target histogram."""
    # Calculate histogram and CDF of the source image
    src_hist, _ = np.histogram(src_img.flatten(), 256, [0,256])
    src_cdf = src_hist.cumsum()
    src_cdf_normalized = src_cdf * src_hist.max() / src_cdf.max() # for display

    # Calculate CDF of the target histogram
    target_cdf = target_hist.cumsum()

    # Create a lookup table (LUT) to map pixel values
    lut = np.zeros(256, dtype=np.uint8)
    g_j = 0
    for g_i in range(256):
        # Find the target pixel value g_j that corresponds to the source pixel value g_i
        while g_j < 255 and src_cdf[g_i] > target_cdf.sum() * src_cdf.max() / target_cdf.max() * target_cdf[g_j]:
             g_j += 1
        lut[g_i] = g_j


    # Apply the LUT to the source image
    matched_img = cv2.LUT(src_img, lut)
    return matched_img

def main():
    # Load the source image in grayscale
    src_path = '/Volumes/Blankspace/DIP and Robot Vision Lab/Fifth lab/Screenshot 2025-11-18 at 3.10.56 PM.png'
    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

    if src_img is None:
        print(f"Error: Could not load image at {src_path}")
        sys.exit(1)

    # 1. Create the target bimodal histogram.
    # These parameters are chosen to create a bimodal distribution based on the example image.
    # You can adjust these values to change the shape of the target histogram.
    mean1, std1 = 80, 25
    mean2, std2 = 170, 25
    target_hist_dist = create_bimodal_gaussian_histogram(mean1, std1, mean2, std2)

    # 2. Perform histogram matching
    output_img = histogram_matching(src_img, target_hist_dist)

    # 3. Save the output image
    output_path = '/Volumes/Blankspace/DIP and Robot Vision Lab/Fifth lab/screenshot_transformed.png'
    cv2.imwrite(output_path, output_img)

    # 4. Generate and save comparison plots
    src_hist_plot = cv2.calcHist([src_img], [0], None, [256], [0, 256])
    output_hist_plot = cv2.calcHist([output_img], [0], None, [256], [0, 256])

    plt.figure(figsize=(18, 12))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(src_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Transformed Image
    plt.subplot(2, 3, 2)
    plt.imshow(output_img, cmap='gray')
    plt.title('Transformed Image')
    plt.axis('off')

    # Bimodal Guass image
    bimodal_guass_path = '/Volumes/Blankspace/DIP and Robot Vision Lab/Fifth lab/bimodal_guass.jpg'
    bimodal_guass_img = cv2.imread(bimodal_guass_path)
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(bimodal_guass_img, cv2.COLOR_BGR2RGB))
    plt.title('Target Histogram Shape')
    plt.axis('off')

    # Original Histogram
    plt.subplot(2, 3, 4)
    plt.title('Original Histogram')
    plt.plot(src_hist_plot, color='b')
    plt.xlim([0, 256])

    # Target Histogram
    plt.subplot(2, 3, 5)
    plt.title('Target Bimodal Histogram')
    # Scale for visual comparison
    plt.plot(target_hist_dist * src_hist_plot.max() / target_hist_dist.max(), color='r')
    plt.xlim([0, 256])

    # Resulting Histogram
    plt.subplot(2, 3, 6)
    plt.title('Resulting Histogram')
    plt.plot(output_hist_plot, color='g')
    plt.xlim([0, 256])


    plt.tight_layout()
    plot_path = '/Volumes/Blankspace/DIP and Robot Vision Lab/Fifth lab/screenshot_comparison_plot.png'
    plt.savefig(plot_path)

    print(f"Transformed image saved to {output_path}")
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
