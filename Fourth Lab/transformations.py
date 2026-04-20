
import cv2
import numpy as np


def apply_gamma(image, gamma: float) -> np.ndarray:
    """Apply gamma correction using a lookup table."""
    # Build lookup table and apply
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(image, table)


def transform_images(input_path: str):
    """
    Applies various transformations to an input image and saves the results.
    Uses OpenCV (cv2) and NumPy instead of Pillow.
    """
    try:
        # Open the input image in grayscale
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"The file '{input_path}' was not found or is not a valid image.")

        # --- Result1: Color Inversion ---
        img_result1 = cv2.bitwise_not(img)
        cv2.imwrite("Result1_generated.png", img_result1)

        # --- Result2: Increased Contrast (Refined) ---
        # Using a simple linear contrast adjustment: result = alpha*img + beta
        alpha = 4.5  # contrast factor (matches previous value)
        beta = 0     # brightness offset
        img_result2 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv2.imwrite("Result2_generated.png", img_result2)

        # --- Result3: Increased Gamma (Brighter - Refined) ---
        gamma_brighter = 0.6  # gamma < 1 brightens
        img_result3 = apply_gamma(img, gamma_brighter)
        cv2.imwrite("Result3_generated.png", img_result3)

        # --- Result4: Decreased Gamma (Darker - Refined) ---
        gamma_darker = 3.0  # gamma > 1 darkens
        img_result4 = apply_gamma(img, gamma_darker)
        cv2.imwrite("Result4_generated.png", img_result4)

        print("All transformations have been applied and saved (OpenCV).")

    except FileNotFoundError as fnf:
        print(fnf)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    transform_images("input.png")
