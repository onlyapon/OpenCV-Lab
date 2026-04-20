
import cv2

def check(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Failed to load {path}")
    else:
        print(f"{path}: {img.shape}")

check("background.webp")
check("foreground.jpeg")
check("4.png")
