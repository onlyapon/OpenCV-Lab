
import cv2
import numpy as np

def composite_images():
    bg_path = "background.webp"
    fg_path = "foreground.jpeg"
    
    bg = cv2.imread(bg_path)
    fg = cv2.imread(fg_path)
    
    if bg is None or fg is None:
        return

    # 1. Resize Background
    target_bg_width = 1200
    ratio_bg = target_bg_width / bg.shape[1]
    new_bg_dim = (target_bg_width, int(bg.shape[0] * ratio_bg))
    bg = cv2.resize(bg, new_bg_dim)
    
    # 2. Resize Foreground relative to Background
    # 25% height of BG seems appropriate for an object on the ground
    target_fg_height = int(bg.shape[0] * 0.25)
    ratio_fg = target_fg_height / fg.shape[0]
    new_fg_dim = (int(fg.shape[1] * ratio_fg), target_fg_height)
    fg = cv2.resize(fg, new_fg_dim)
    
    rows, cols, channels = fg.shape
    print(f"BG Size: {bg.shape}, FG Size: {fg.shape}")

    # 3. Create Mask
    # Threshold light background (>200) to strip it.
    # Using a slightly higher threshold to ensure we don't accidentally crop bright parts of the object
    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(fg_gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 3) 
    mask_inv = cv2.bitwise_not(mask)
    
    # 4. ROI Placement: Bottom-Right corner
    # Place it 50 pixels from bottom and 50 pixels from right edge
    y_offset = bg.shape[0] - rows - 50
    x_offset = bg.shape[1] - cols - 50
    
    # Safety check
    if y_offset < 0: y_offset = 0
    if x_offset < 0: x_offset = 0
    
    print(f"Placing at x={x_offset}, y={y_offset}")

    roi = bg[y_offset:y_offset+rows, x_offset:x_offset+cols]
    
    # 5. Merge
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(fg, fg, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    bg[y_offset:y_offset+rows, x_offset:x_offset+cols] = dst
    
    cv2.imwrite("task4_output.png", bg)
    print("Success: task4_output.png saved.")

if __name__ == "__main__":
    composite_images()
