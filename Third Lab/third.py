import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import affine_transform

# =============================================================================
# Task 1: Interpolation from Scratch (Bilinear & Bicubic)
# =============================================================================

def bilinear_resize(image, new_height, new_width):
    """
    Resizes an image using bilinear interpolation.
    """
    old_height, old_width, channels = image.shape
    new_image = np.zeros((new_height, new_width, channels))

    # Calculate scaling ratios
    y_ratio = old_height / new_height
    x_ratio = old_width / new_width

    for y_new in range(new_height):
        for x_new in range(new_width):
            # Find corresponding coordinates in the original image
            y_orig = y_new * y_ratio
            x_orig = x_new * x_ratio

            # Get integer and fractional parts
            y1 = int(np.floor(y_orig))
            x1 = int(np.floor(x_orig))
            
            # Ensure coordinates are within bounds
            y2 = min(y1 + 1, old_height - 1)
            x2 = min(x1 + 1, old_width - 1)
            y1 = max(0, y1)
            x1 = max(0, x1)

            # Get fractional distances
            dy = y_orig - y1
            dx = x_orig - x1

            # Get the 4 neighboring pixels
            P1 = image[y1, x1]
            P2 = image[y1, x2]
            P3 = image[y2, x1]
            P4 = image[y2, x2]

            # Interpolate in x-direction
            R1 = (1 - dx) * P1 + dx * P2
            R2 = (1 - dx) * P3 + dx * P4

            # Interpolate in y-direction
            new_pixel = (1 - dy) * R1 + dy * R2
            
            new_image[y_new, x_new] = new_pixel

    return new_image.astype(np.uint8)

def cubic_kernel(s, a=-0.5):
    """
    Standard bicubic kernel (Catmull-Rom spline).
    'a' is often -0.5 or -0.75.
    """
    s = np.abs(s)
    if s <= 1:
        return (a + 2) * s**3 - (a + 3) * s**2 + 1
    elif 1 < s <= 2:
        return a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a
    return 0

def bicubic_resize(image, new_height, new_width):
    """
    Resizes an image using bicubic interpolation.
    """
    old_height, old_width, channels = image.shape
    new_image = np.zeros((new_height, new_width, channels))

    y_ratio = old_height / new_height
    x_ratio = old_width / new_width

    for y_new in range(new_height):
        for x_new in range(new_width):
            
            y_orig = y_new * y_ratio
            x_orig = x_new * x_ratio

            y1_int = int(np.floor(y_orig))
            x1_int = int(np.floor(x_orig))
            
            new_pixel = np.zeros(channels)
            
            # Loop over the 4x4 neighborhood
            for j in range(-1, 3): # y-neighborhood
                for i in range(-1, 3): # x-neighborhood
                    
                    # Get pixel coordinates, clamping to edges
                    y_idx = np.clip(y1_int + j, 0, old_height - 1)
                    x_idx = np.clip(x1_int + i, 0, old_width - 1)
                    
                    # Get the pixel value
                    P = image[y_idx, x_idx]
                    
                    # Calculate weights
                    weight_y = cubic_kernel(y_orig - (y1_int + j))
                    weight_x = cubic_kernel(x_orig - (x1_int + i))
                    
                    # Add to the weighted sum
                    new_pixel += P * weight_x * weight_y

            new_image[y_new, x_new] = new_pixel

    # Clip values to valid range [0, 255]
    return np.clip(new_image, 0, 255).astype(np.uint8)


# =============================================================================
# Task 2: Affine Transformations
# =============================================================================

def apply_transform(image, matrix, output_shape=None):
    """
    Applies an affine transformation using scipy.
    The matrix should be the INVERSE matrix (output-to-input mapping).
    """
    if output_shape is None:
        output_shape = image.shape
        
    # Separate the linear part (matrix) and translation (offset)
    transform_matrix = matrix[:2, :2]
    offset = matrix[:2, 2]
    
    # Apply the transform to each channel separately
    transformed_channels = []
    for c in range(image.shape[2]):
        transformed_channel = affine_transform(
            image[:, :, c],
            transform_matrix,
            offset=offset,
            output_shape=output_shape[:2],
            order=1, # 1 = Bilinear interpolation
            cval=0.0 # Fill value for pixels outside bounds
        )
        transformed_channels.append(transformed_channel)
        
    # Stack channels back together
    return np.stack(transformed_channels, axis=-1).astype(np.uint8)

# =============================================================================
# Main Execution (Loading, Processing, Plotting)
# =============================================================================

# --- Load Image ---
try:
    img = mpimg.imread('/Volumes/Blankspace/DIP and Robot Vision Lab/Third Lab/monalisa.jpg')
except FileNotFoundError:
    print("Error: 'monalisa.jpg' not found. Please place it in the same directory.")
    exit()

h, w, c = img.shape
print(f"Original image loaded: {w}x{h}")

# --- Task 1: Resizing ---
print("Applying Bilinear resizing...")
new_h, new_w = int(h * 1.5), int(w * 1.5)
bilinear_img = bilinear_resize(img, new_h, new_w)

print("Applying Bicubic resizing...")
bicubic_img = bicubic_resize(img, new_h, new_w)


# --- Task 2: Affine Transformations ---
print("Applying Affine transformations...")

# Center for rotation/scaling (row, col)
center = np.array([h / 2, w / 2])
center_h = np.array([center[0], center[1], 1])

# 1. Translation
tr, tc = 80, 50 # Translate 80 rows (down), 50 cols (right)
M_trans_fwd = np.array([[1, 0, tr],
                        [0, 1, tc],
                        [0, 0, 1]])
M_trans_inv = np.linalg.inv(M_trans_fwd)
translated_img = apply_transform(img, M_trans_inv)


# 2. Scaling (about center)
sr, sc = 0.8, 1.2 # Scale 0.8 in row, 1.2 in col
T_neg = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
S = np.array([[sr, 0, 0], [0, sc, 0], [0, 0, 1]])
T_pos = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])
M_scale_fwd = T_pos @ S @ T_neg
M_scale_inv = np.linalg.inv(M_scale_fwd)
scaled_img = apply_transform(img, M_scale_inv)


# 3. Rotation (about center)
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) # Note: (row, col) rotation matrix
M_rot_fwd = T_pos @ R @ T_neg
M_rot_inv = np.linalg.inv(M_rot_fwd)
rotated_img = apply_transform(img, M_rot_inv)


# 4. Shear (about center)
k = 0.2 # Shear factor for columns based on row
K = np.array([[1, 0, 0], [k, 1, 0], [0, 0, 1]])
M_shear_fwd = T_pos @ K @ T_neg
M_shear_inv = np.linalg.inv(M_shear_fwd)
sheared_img = apply_transform(img, M_shear_inv)


# --- Display Results ---
print("Displaying results...")
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Row 1
axs[0, 0].imshow(img)
axs[0, 0].set_title(f"Original ({w}x{h})")
axs[0, 0].axis('off')

axs[0, 1].axis('off') # Empty
axs[0, 2].axis('off') # Empty

# Row 2
axs[1, 0].imshow(bilinear_img)
axs[1, 0].set_title(f"Bilinear Resize ({new_w}x{new_h})")
axs[1, 0].axis('off')

axs[1, 1].imshow(bicubic_img)
axs[1, 1].set_title(f"Bicubic Resize ({new_w}x{new_h})")
axs[1, 1].axis('off')

axs[1, 2].imshow(translated_img)
axs[1, 2].set_title(f"Translated (r_c+{tr}, c_c+{tc})")
axs[1, 2].axis('off')

# Row 3
axs[2, 0].imshow(rotated_img)
axs[2, 0].set_title("Rotated (30 deg)")
axs[2, 0].axis('off')

axs[2, 1].imshow(scaled_img)
axs[2, 1].set_title(f"Scaled (s_r={sr}, s_c={sc})")
axs[2, 1].axis('off')

axs[2, 2].imshow(sheared_img)
axs[2, 2].set_title(f"Sheared (k={k})")
axs[2, 2].axis('off')

plt.tight_layout()
plt.suptitle("Computer Vision Lab: Interpolation and Affine Transforms", fontsize=16, y=1.02)
plt.show()