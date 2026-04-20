
import cv2
import numpy as np

def analyze_corners(path):
    img = cv2.imread(path)
    if img is None:
        return
    
    # Check corners: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    corners = [
        img[0, 0],
        img[0, -1],
        img[-1, 0],
        img[-1, -1]
    ]
    
    print(f"Corner colors (BGR) for {path}:")
    for i, c in enumerate(corners):
        print(f"  Corner {i}: {c}")
        
    # Average color of top row to guess background
    avg_top = np.mean(img[0, :], axis=0)
    print(f"  Average Top Row: {avg_top}")

analyze_corners("foreground.jpeg")
