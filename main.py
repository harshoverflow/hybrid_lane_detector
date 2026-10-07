import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# =========================================================================
# 1. LOAD PRETRAINED SEGFORMER (Pixel-Wise Classifier)
# =========================================================================
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

print("Loading SegFormer model...")
feature_extractor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()
print("Model ready.\n")


def select_input_image(directory="test_images"):
    os.makedirs(directory, exist_ok=True)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported_extensions)]

    print("=" * 55)
    print(f"       SELECT AN IMAGE FROM '{directory}/'")
    print("=" * 55)
    for i, file_name in enumerate(files, 1):
        print(f" {i}. {file_name}")
    print(" 0. Browse any file from computer (Open File Dialog)")
    print("=" * 55)

    choice = input(f"Enter choice (0-{len(files)}): ").strip()
    if choice == "0":
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Road Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All Files", "*.*")]
        )
        root.destroy()
        return path
    elif choice.isdigit() and 1 <= int(choice) <= len(files):
        return os.path.join(directory, files[int(choice) - 1])
    elif len(files) > 0:
        return os.path.join(directory, files[0])
    return None


def traditional_cv_preprocess(image_rgb):
    """Enhance local contrast with CLAHE and smooth pavement noise."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)


def ml_pixel_wise_classification(image_rgb):
    """Pixel-wise drivable road surface mask from SegFormer."""
    inputs = feature_extractor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled = torch.nn.functional.interpolate(
        logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy()
    return (pred_seg == 0).astype(np.uint8) * 255


# =========================================================================
# 4. ROBUST STRIPE EXTRACTION (GLARE-PROOF VIA TOP-HAT FILTERING)
# =========================================================================
def extract_slender_lane_stripes(image_rgb, road_mask):
    """
    Extracts lane markings by using morphological Top-Hat filtering.
    A horizontal kernel isolates thin bright vertical stripes (lanes)
    and completely eliminates wide glare/reflection puddles.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Horizontal White Top-Hat: preserves elements narrower than 28 pixels
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

    # 2. Also capture distinct yellow/orange paint in HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([38, 255, 255]))

    # Threshold the top-hat response
    _, tophat_thresh = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)

    # Combine thin white stripes + yellow lines
    combined_stripes = cv2.bitwise_or(tophat_thresh, yellow_mask)

    # Confine strictly within ML Drivable Road Surface
    lane_pixels = cv2.bitwise_and(combined_stripes, road_mask)

    # Remove isolated salt-and-pepper noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_lanes = cv2.morphologyEx(lane_pixels, cv2.MORPH_OPEN, kernel_clean)
    return clean_lanes


# =========================================================================
# 5. VANISHING-POINT-CONSTRAINED MULTI-LANE TRACKING
# =========================================================================
def fit_robust_lane_lines(image_rgb, clean_lanes, road_mask):
    """
    Detects line segments, filters out non-road slopes, and clusters them
    into stable continuous lane tracks converging toward the vanishing zone.
    """
    h, w = clean_lanes.shape

    # Find horizon (topmost boundary of road mask)
    road_y_coords = np.nonzero(road_mask)[0]
    y_horizon = int(np.percentile(road_y_coords, 2)) if len(road_y_coords) > 0 else int(h * 0.52)
    y_horizon = max(int(h * 0.45), min(y_horizon, int(h * 0.65)))
    y_bottom = h

    # Detect line segments using HoughLinesP
    lines = cv2.HoughLinesP(
        clean_lanes, rho=1, theta=np.pi / 180, threshold=15,
        minLineLength=18, maxLineGap=35
    )

    line_canvas = np.zeros_like(image_rgb)
    overlay = image_rgb.copy()
    # Drivable road carpet overlay in translucent green
    overlay[road_mask > 0] = overlay[road_mask > 0] * 0.78 + np.array([0, 180, 0], dtype=np.uint8) * 0.22

    if lines is None:
        return overlay

    # Filter lines: only keep segments that point upward toward the vanishing area
    valid_segments = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)
        if y1 == y2:
            continue
        # Ensure y1 is bottom, y2 is top
        if y1 < y2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        dx = x2 - x1
        dy = y2 - y1  # Negative because y2 < y1
        slope = dy / dx if dx != 0 else -999.0

        # Discard horizontal noise (abs slope < 0.35)
        if abs(slope) < 0.35:
            continue

        # Extrapolate x to the bottom edge of the image
        x_bottom = int(x1 + (y_bottom - y1) / slope)

        # Only retain lines whose base lies reasonably within or near road width
        if -w * 0.2 <= x_bottom <= w * 1.2:
            valid_segments.append((x_bottom, x1, y1, x2, y2))

    if len(valid_segments) == 0:
        return overlay

    # Cluster segments by bottom x-coordinate (within 65-pixel corridors)
    valid_segments.sort(key=lambda s: s[0])
    clusters = []
    current_cluster = [valid_segments[0]]

    for seg in valid_segments[1:]:
        if abs(seg[0] - current_cluster[-1][0]) < 65:
            current_cluster.append(seg)
        else:
            clusters.append(current_cluster)
            current_cluster = [seg]
    clusters.append(current_cluster)

    # Fit and draw each verified lane line
    for cluster in clusters:
        # Require enough evidence (at least 2 segments or total segment length > 40px)
        total_len = sum(np.hypot(s[3] - s[1], s[4] - s[2]) for s in cluster)
        if len(cluster) >= 2 or total_len > 40:
            xs = [pt for s in cluster for pt in (s[1], s[3])]
            ys = [pt for s in cluster for pt in (s[2], s[4])]

            # Robust linear fit x = m*y + b
            fit = np.polyfit(ys, xs, 1)
            x_bot = int(fit[0] * y_bottom + fit[1])
            x_top = int(fit[0] * y_horizon + fit[1])

            # Draw smooth continuous red lane marking
            cv2.line(line_canvas, (x_bot, y_bottom), (x_top, y_horizon), (255, 0, 0), 5)
        else:
            # Draw individual segment
            for s in cluster:
                cv2.line(line_canvas, (s[1], s[2]), (s[3], s[4]), (255, 0, 0), 4)

    overlay[line_canvas[:, :, 0] > 0] = [255, 0, 0]
    return overlay


# =========================================================================
# 6. RUN
# =========================================================================
if __name__ == "__main__":
    image_path = select_input_image("test_images")
    if not image_path or not os.path.exists(image_path):
        print("No image selected.")
        exit(0)

    print(f"\nProcessing: {image_path}")
    raw_bgr = cv2.imread(image_path)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    print("Stage 1: Pre-processing...")
    preprocessed_img = traditional_cv_preprocess(raw_rgb)

    print("Stage 2: ML Drivable Road Surface Segmentation...")
    road_mask = ml_pixel_wise_classification(preprocessed_img)

    print("Stage 3: Morphological Top-Hat Glare Filter...")
    clean_lanes = extract_slender_lane_stripes(raw_rgb, road_mask)

    print("Stage 4: Robust Vanishing-Point Line Tracking...")
    final_overlay = fit_robust_lane_lines(raw_rgb, clean_lanes, road_mask)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title(f"1. Input ({os.path.basename(image_path)})", fontweight="bold")

    axes[0, 1].imshow(road_mask, cmap="gray")
    axes[0, 1].set_title("2. ML: Drivable Surface Mask", fontweight="bold")

    axes[1, 0].imshow(clean_lanes, cmap="gray")
    axes[1, 0].set_title("3. Top-Hat Filter (Glare Eliminated)", fontweight="bold")

    axes[1, 1].imshow(final_overlay)
    axes[1, 1].set_title("4. Multi-Lane Tracking Overlay", fontweight="bold")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    print("Done!")