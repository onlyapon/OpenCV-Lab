# Digital Image Processing and Robot Vision Lab

Coursework and lab assignments for the Digital Image Processing (DIP) and Robot Vision course. Each folder contains the Python implementation, input/output images, and (where applicable) a lab report for an individual lab.

**Author:** Ajmain Istiak Apon (FH-172-011)

## Requirements

- Python 3.x
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-image`
- `imageio`

Install with:

```bash
pip install opencv-python numpy matplotlib scipy scikit-image imageio
```

## Labs

### First Lab — Histograms, Equalization, and Thresholding
- `first_assignment.py` — loads two images, computes grayscale histograms, applies histogram equalization, CLAHE, global thresholding (binary, inverse, to-zero, truncated, Otsu), and adaptive thresholding (mean-C and Gaussian-C).

### Second Lab — Grayscale Histogram Analysis
- `labtask.py` — loads five images, converts them to grayscale, and computes their histograms for visual comparison.

### Third Lab — Interpolation and Affine Transformations
- `third.py` — implements bilinear and bicubic interpolation from scratch, and applies translation, scaling, rotation, and shear using affine matrices (via `scipy.ndimage.affine_transform`).

### Fourth Lab — Intensity Transformations
- `transformations.py` — grayscale intensity transforms on a test image: color inversion, linear contrast stretching, and gamma correction (brighter and darker variants).
- Includes a LaTeX lab report (`lab_report.tex`, `lab_report.pdf`).

### Fifth Lab — Histogram Matching
- `histogram_matching.py` — histogram specification against a bimodal Gaussian target distribution using CDF mapping.
- `plot_histogram.py` — histogram plotting helper.
- Reports: `report.md`, `lab_report.md`, and PDF report.

### Sixth Lab — Histogram Equalization Variants
- `he.py` — standard global histogram equalization (grayscale and per-channel RGB) using `skimage.exposure`.
- `dhe.py` — dynamic histogram equalization operating in the HSV colour space with gradient-based analysis.

### Seventh Lab — Noise Removal and Image Compositing
- `lab.py` — salt-and-pepper noise removal via median filtering.
- `task3.py` — Gaussian blur, bilateral filter, and non-local means denoising comparison.
- `task4.py` / `match_task4.py` — foreground/background compositing.
- `analyze_color.py`, `analyze_noise.py`, `check_img.py`, `check_task4.py` — supporting analysis scripts.

### Lab 8 — Canny Edge Detection
- `task1_canny.py` — Canny edge detection from scratch (Gaussian smoothing, Sobel gradients, non-maximum suppression, double thresholding, hysteresis).
- `task2_modifications.py` — variations on the baseline Canny pipeline (modified gradient angles, NMS ablation, parameter tuning).
- Includes a markdown and PDF lab report.

## Running

Each lab is self-contained. From within a lab's folder:

```bash
cd "First Lab"
python first_assignment.py
```

Most scripts expect image inputs to be present in the same directory. A few use absolute paths and may need the path updated for your machine.
