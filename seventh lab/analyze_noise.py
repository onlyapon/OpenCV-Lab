
import cv2
import numpy as np

def analyze_noise(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return

    # Split into Left (Noisy) and Right (Clean/Target)
    height, width, _ = img.shape
    midpoint = width // 2
    noisy_img = img[:, :midpoint]
    target_img = img[:, midpoint:]
    
    # Resize target to match noisy if off by a pixel (odd widths)
    target_img = cv2.resize(target_img, (noisy_img.shape[1], noisy_img.shape[0]))

    print(f"Image loaded. Split into Noisy {noisy_img.shape} and Target {target_img.shape}")

    # Convert to grayscale for simple MSE calculation
    noisy_gray = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    results = {}

    # 1. Median Filter (Good for Salt & Pepper)
    for k in [3, 5, 7]:
        denoised = cv2.medianBlur(noisy_gray, k)
        mse = np.mean((target_gray - denoised) ** 2)
        results[f"Median {k}x{k}"] = mse

    # 2. Gaussian Blur (Good for Gaussian Noise)
    for k in [3, 5, 7]:
        denoised = cv2.GaussianBlur(noisy_gray, (k, k), 0)
        mse = np.mean((target_gray - denoised) ** 2)
        results[f"Gaussian {k}x{k}"] = mse

    # 3. Mean Filter / Box Filter
    for k in [3, 5, 7]:
        denoised = cv2.blur(noisy_gray, (k, k))
        mse = np.mean((target_gray - denoised) ** 2)
        results[f"Mean/Box {k}x{k}"] = mse
        
    # Print results sorted by MSE (lower is better)
    print("\n--- MSE Results (Lower is Better) ---")
    sorted_results = sorted(results.items(), key=lambda item: item[1])
    for name, mse in sorted_results:
        print(f"{name}: {mse:.2f}")

    best_method = sorted_results[0][0]
    print(f"\nBest match: {best_method}")
    
    if "Median" in best_method:
        print("Diagnosis: Salt and Pepper Noise (Impulse Noise)")
    elif "Gaussian" in best_method or "Mean" in best_method:
        print("Diagnosis: Gaussian Noise or similar random noise")

if __name__ == "__main__":
    analyze_noise("2.png")
