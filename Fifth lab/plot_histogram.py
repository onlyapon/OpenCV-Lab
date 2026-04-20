
import cv2
import matplotlib.pyplot as plt
import sys

def plot_histogram(image_path, output_path):
    """
    Loads an image, calculates its histogram, and saves the plot.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
        sys.exit(1)

    # Calculate the histogram
    # hist is a 256x1 array, where each entry corresponds to the number of pixels for that intensity value
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Create a plot for the histogram
    plt.figure(figsize=(10, 6))
    plt.title("Bimodal Histogram of Transformed Image")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Number of Pixels")
    plt.plot(hist, color='darkblue')
    plt.xlim([0, 256])
    plt.grid(axis='y', alpha=0.75)

    # Save the histogram plot to a file
    plt.savefig(output_path)
    print(f"Histogram plot saved to {output_path}")

if __name__ == "__main__":
    # Define the path to your transformed image
    image_to_process = "/Volumes/Blankspace/DIP and Robot Vision Lab/Fifth lab/phobos_transformed.png"
    
    # Define where to save the output histogram plot
    output_plot_path = "/Volumes/Blankspace/DIP and Robot Vision Lab/Fifth lab/phobos_transformed_histogram.png"
    
    # Generate and save the histogram
    plot_histogram(image_to_process, output_plot_path)
