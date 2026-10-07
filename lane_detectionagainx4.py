import os
import re
import csv
import time
import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from transformers import SegformerForSemanticSegmentation

# =========================================================================
# 1. LOAD PRETRAINED SEGFORMER (Pixel-Wise Road Classifier)
# =========================================================================
# B0 is the fast default. For slightly better road edges (about 2-3x slower) use:
#   "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
SEG_INPUT_SIZE = (1024, 512)   # (width, height) the road model sees; smaller = faster

TEST_DIR = "test_images"
RESULTS_DIR = "results"
LABELS_FILE = os.path.join(TEST_DIR, "labels.csv")  # format: filename,true_lanes
SUPPORTED = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MAX_WIDTH = 1024  # resize large photos so filter sizes stay consistent

print("Loading SegFormer model...")
torch.set_num_threads(os.cpu_count() or 4)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
print("SegFormer ready.\n")

# ---- Trained U-Net (TuSimple lane markings, input 256x512, sigmoid output) ----
UNET_PATH = "tusimple_lane_unet_bestever.keras"   # put this file next to the script
UNET_THRESHOLD = 0.8   # best on the TuSimple evaluation (mode 4)
unet_model = None
if os.path.exists(UNET_PATH):
    try:
        import keras
        unet_model = keras.models.load_model(UNET_PATH, compile=False)
        print(f"Trained U-Net loaded: {UNET_PATH}\n")
    except Exception as e:
        print(f"Could not load the U-Net ({e}). Falling back to the top-hat filter.\n")
else:
    print(f"'{UNET_PATH}' not found next to the script. Falling back to the top-hat filter.\n")


# =========================================================================
# 2. IMAGE LOADING / SELECTION
# =========================================================================
def load_image_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        rgb = cv2.resize(rgb, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def list_test_images(directory=TEST_DIR):
    os.makedirs(directory, exist_ok=True)
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(SUPPORTED))


def select_single_image(directory=TEST_DIR):
    files = list_test_images(directory)
    print("=" * 55)
    print(f"       SELECT AN IMAGE FROM '{directory}/'")
    print("=" * 55)
    for i, name in enumerate(files, 1):
        print(f" {i}. {name}")
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
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return os.path.join(directory, files[int(choice) - 1])
    if files:
        return os.path.join(directory, files[0])
    return None


def load_ground_truth(labels_file=LABELS_FILE):
    """Reads labels.csv (filename,true_lanes). Returns {filename: int}."""
    truth = {}
    if not os.path.exists(labels_file):
        return truth
    with open(labels_file, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            name, value = row[0].strip(), row[1].strip()
            if value.isdigit():
                truth[name] = int(value)
    return truth


# =========================================================================
# 3. TRADITIONAL CV PREPROCESSING + ML ROAD SEGMENTATION
# =========================================================================
def traditional_cv_preprocess(image_rgb):
    """Enhance local contrast with CLAHE and smooth pavement noise."""
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)


def ml_pixel_wise_classification(image_rgb):
    """Pixel-wise drivable road mask from SegFormer (Cityscapes class 0 = road)."""
    h, w = image_rgb.shape[:2]
    small = cv2.resize(image_rgb, SEG_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    x = (small.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    with torch.inference_mode():
        logits = model(pixel_values=x).logits[0]               # (19, H/4, W/4)
    # Road wins where its logit beats every other class. One margin map is far cheaper
    # to upsample than all 19 class maps.
    margin = (logits[0] - logits[1:].max(dim=0).values).cpu().numpy()
    margin = cv2.resize(margin, (w, h), interpolation=cv2.INTER_LINEAR)
    return ((margin > 0) * 255).astype(np.uint8)


# =========================================================================
# 4. STRIPE EXTRACTION (GLARE-PROOF VIA TOP-HAT FILTERING)
# =========================================================================
def extract_slender_lane_stripes(image_rgb, road_mask):
    """Thin bright stripes via horizontal top-hat + yellow paint, confined to the road mask."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([38, 255, 255]))

    _, tophat_thresh = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(tophat_thresh, yellow_mask)
    lane_pixels = cv2.bitwise_and(combined, road_mask)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(lane_pixels, cv2.MORPH_OPEN, kernel_clean)


# =========================================================================
# 5. LINE TRACKING + LANE COUNTING
# =========================================================================
def fit_robust_lane_lines(image_rgb, clean_lanes, road_mask):
    """
    Detects line segments, clusters them into lane boundary lines, and counts lanes.
    Returns (overlay_image, num_boundary_lines, num_lanes).
    Lanes = boundary lines - 1 (lines are the edges, lanes are the gaps between them).
    """
    h, w = clean_lanes.shape

    road_y_coords = np.nonzero(road_mask)[0]
    y_horizon = int(np.percentile(road_y_coords, 2)) if len(road_y_coords) > 0 else int(h * 0.52)
    y_horizon = max(int(h * 0.45), min(y_horizon, int(h * 0.65)))
    y_bottom = h

    lines = cv2.HoughLinesP(
        clean_lanes, rho=1, theta=np.pi / 180, threshold=15,
        minLineLength=18, maxLineGap=35
    )

    line_canvas = np.zeros_like(image_rgb)
    overlay = image_rgb.copy()
    overlay[road_mask > 0] = (overlay[road_mask > 0] * 0.78 +
                              np.array([0, 180, 0], dtype=np.float32) * 0.22).astype(np.uint8)

    if lines is None:
        return overlay, 0, 0

    valid_segments = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(-1)
        if y1 == y2:
            continue
        if y1 < y2:  # make (x1, y1) the bottom point
            x1, y1, x2, y2 = x2, y2, x1, y1

        dx = x2 - x1
        dy = y2 - y1
        slope = dy / dx if dx != 0 else -999.0
        if abs(slope) < 0.35:
            continue

        x_bottom = int(x1 + (y_bottom - y1) / slope)
        if -w * 0.2 <= x_bottom <= w * 1.2:
            valid_segments.append((x_bottom, x1, y1, x2, y2))

    if not valid_segments:
        return overlay, 0, 0

    valid_segments.sort(key=lambda s: s[0])
    clusters = []
    current = [valid_segments[0]]
    for seg in valid_segments[1:]:
        if abs(seg[0] - current[-1][0]) < 65:
            current.append(seg)
        else:
            clusters.append(current)
            current = [seg]
    clusters.append(current)

    verified_lines = 0
    for cluster in clusters:
        total_len = sum(np.hypot(s[3] - s[1], s[4] - s[2]) for s in cluster)
        # Only clusters with enough evidence count as a real lane boundary
        if len(cluster) >= 2 or total_len > 40:
            xs = [pt for s in cluster for pt in (s[1], s[3])]
            ys = [pt for s in cluster for pt in (s[2], s[4])]
            if len(set(ys)) < 2:
                continue
            fit = np.polyfit(ys, xs, 1)
            x_bot = int(fit[0] * y_bottom + fit[1])
            x_top = int(fit[0] * y_horizon + fit[1])
            cv2.line(line_canvas, (x_bot, y_bottom), (x_top, y_horizon), (255, 0, 0), 5)
            verified_lines += 1

    overlay[line_canvas[:, :, 0] > 0] = [255, 0, 0]
    num_lanes = max(verified_lines - 1, 0)
    return overlay, verified_lines, num_lanes


# =========================================================================
# 6. FULL PIPELINE ON ONE IMAGE
# =========================================================================
def ml_lane_marking_mask(image_rgb):
    """Trained U-Net: per-pixel probability of 'lane marking', thresholded to a binary mask."""
    h, w = image_rgb.shape[:2]
    x = cv2.resize(image_rgb, (512, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    prob = unet_model(x[None], training=False).numpy()[0, :, :, 0]
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return ((prob > UNET_THRESHOLD) * 255).astype(np.uint8)


def remove_small_blobs(mask, min_area):
    """Drops connected components smaller than min_area pixels (speckle noise)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep_ids = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep_ids[0] = False                      # label 0 is the background
    return (keep_ids[labels] * 255).astype(np.uint8)


def run_pipeline(image_rgb):
    t = time.time()

    def tick(label):
        nonlocal t
        now = time.time()
        print(f"     {label}: {now - t:.1f}s", flush=True)
        t = now

    preprocessed = traditional_cv_preprocess(image_rgb)          # traditional CV
    tick("preprocessing")
    road_mask = ml_pixel_wise_classification(preprocessed)       # ML #1: SegFormer road
    tick("SegFormer road mask")

    if unet_model is not None:
        lane_mask = ml_lane_marking_mask(image_rgb)              # ML #2: trained U-Net lanes (raw input, matches training)
        tick("U-Net lane mask")
        lane_pixels = cv2.bitwise_and(lane_mask, road_mask)      # keep lane pixels on the road only
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lane_pixels = cv2.morphologyEx(lane_pixels, cv2.MORPH_OPEN, kernel)
        min_area = max(30, int(image_rgb.shape[0] * image_rgb.shape[1] * 0.0002))
        clean_lanes = remove_small_blobs(lane_pixels, min_area)
        lane_title = "3. U-Net lane markings AND road mask"
    else:
        clean_lanes = extract_slender_lane_stripes(image_rgb, road_mask)
        lane_title = "3. Top-Hat Filter (U-Net not loaded)"

    overlay, n_lines, n_lanes = fit_robust_lane_lines(image_rgb, clean_lanes, road_mask)
    tick("lane fitting")
    return {
        "road_mask": road_mask,
        "clean_lanes": clean_lanes,
        "overlay": overlay,
        "num_lines": n_lines,
        "num_lanes": n_lanes,
        "lane_title": lane_title,
    }


# =========================================================================
# 7. RESULT FIGURES
# =========================================================================
def save_result_figure(image_rgb, result, name, true_lanes=None, out_dir=RESULTS_DIR, show=False):
    os.makedirs(out_dir, exist_ok=True)
    predicted = result["num_lanes"]

    if true_lanes is None:
        verdict, color = f"Predicted lanes: {predicted}", "black"
    elif predicted == true_lanes:
        verdict, color = f"Predicted: {predicted} | True: {true_lanes} | CORRECT", "green"
    else:
        verdict, color = f"Predicted: {predicted} | True: {true_lanes} | WRONG", "red"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title(f"1. Input ({name})", fontweight="bold")
    axes[0, 1].imshow(result["road_mask"], cmap="gray")
    axes[0, 1].set_title("2. ML: Drivable Surface Mask (SegFormer-B1)", fontweight="bold")
    axes[1, 0].imshow(result["clean_lanes"], cmap="gray")
    axes[1, 0].set_title(result.get("lane_title", "3. Lane pixels"), fontweight="bold")
    axes[1, 1].imshow(result["overlay"])
    axes[1, 1].set_title("4. Lane Tracking Overlay", fontweight="bold")
    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(verdict, fontsize=16, fontweight="bold", color=color)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, os.path.splitext(name)[0] + "_result.png")
    fig.savefig(out_path, dpi=90)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# =========================================================================
# 8. ACCURACY EVALUATION
# =========================================================================
def compute_accuracy(rows):
    """rows: list of (name, predicted, true). Returns metrics dict for labeled rows."""
    labeled = [(n, p, t) for n, p, t in rows if t is not None]
    if not labeled:
        return None
    errors = [abs(p - t) for _, p, t in labeled]
    n = len(labeled)
    return {
        "labeled_images": n,
        "exact_accuracy": 100.0 * sum(e == 0 for e in errors) / n,
        "within_one_accuracy": 100.0 * sum(e <= 1 for e in errors) / n,
        "mean_abs_error": sum(errors) / n,
    }


def print_accuracy_report(rows):
    print("\n" + "=" * 60)
    print(f"{'Image':30s} {'Predicted':>10s} {'True':>6s} {'Result':>8s}")
    print("-" * 60)
    for name, pred, true in rows:
        if true is None:
            status, true_s = "n/a", "-"
        else:
            status, true_s = ("OK" if pred == true else "WRONG"), str(true)
        print(f"{name[:30]:30s} {pred:>10d} {true_s:>6s} {status:>8s}")
    print("=" * 60)

    metrics = compute_accuracy(rows)
    if metrics is None:
        print("No ground-truth labels found, so accuracy cannot be computed.")
        print(f"Add them to {LABELS_FILE} as 'filename,true_lanes' lines.")
        return
    print(f"Labeled images        : {metrics['labeled_images']}")
    print(f"Exact-match accuracy  : {metrics['exact_accuracy']:.1f}%")
    print(f"Within +/-1 lane      : {metrics['within_one_accuracy']:.1f}%")
    print(f"Mean absolute error   : {metrics['mean_abs_error']:.2f} lanes")


def save_summary_csv(rows, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_lanes", "true_lanes", "correct"])
        for name, pred, true in rows:
            correct = "" if true is None else int(pred == true)
            writer.writerow([name, pred, "" if true is None else true, correct])
    return path


# =========================================================================
# 9. RUN MODES
# =========================================================================
# =========================================================================
# 8b. KITTI ROAD EVALUATION (pixel-wise ML accuracy vs ground-truth masks)
# =========================================================================
def _name_key(filename):
    """'um_lane_000001.png' -> ('um', '000001'); 'um_000001.png' -> ('um', '000001')."""
    stem = os.path.splitext(filename)[0].lower()
    digits = re.findall(r"\d+", stem)
    if not digits:
        return None
    num = str(int(digits[-1]))  # drop leading zeros so 000001 == 1
    m = re.match(r"[a-z]+", stem)
    prefix = m.group(0) if m else ""
    if prefix in ("image", "gt", "img", "road", "lane"):
        prefix = ""
    return prefix, num


def build_gt_index(gt_dir):
    """Maps (prefix, number) -> list of (path, kind) for every file in gt_image_2."""
    index = {}
    for f in os.listdir(gt_dir):
        if not f.lower().endswith(SUPPORTED):
            continue
        key = _name_key(f)
        if key is None:
            continue
        low = f.lower()
        kind = "road" if "road" in low else "lane"
        index.setdefault(key, []).append((os.path.join(gt_dir, f), kind))
    return index


def find_kitti_gt(gt_index, image_name):
    """Finds the ground-truth mask for an image, preferring *_road_* over *_lane_* masks."""
    key = _name_key(image_name)
    if key is None:
        return None, None
    prefix, num = key
    candidates = gt_index.get((prefix, num)) or gt_index.get(("", num)) or []
    if not candidates:
        return None, None
    candidates = sorted(candidates, key=lambda c: 0 if c[1] == "road" else 1)
    return candidates[0]


def load_kitti_gt_road(path, kind, shape_hw):
    """
    Colour-coded KITTI mask -> boolean road mask.
    magenta = road (or ego lane in *_lane_* files), black = other road in lane files,
    red = not road.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    magenta = (r > 180) & (g < 100) & (b > 180)
    black = (r < 60) & (g < 60) & (b < 60)
    road = (magenta | black) if kind == "lane" else magenta
    road = cv2.resize(road.astype(np.uint8), (shape_hw[1], shape_hw[0]),
                      interpolation=cv2.INTER_NEAREST)
    return road.astype(bool)


def seg_metrics(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    iou = tp / (tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return iou, precision, recall, f1


def save_kitti_figure(image_rgb, pred, gt, name, metrics, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Input ({name})", fontweight="bold")
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("ML road mask (SegFormer-B1)", fontweight="bold")
    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("Ground truth road", fontweight="bold")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"IoU {metrics[0]:.3f} | Precision {metrics[1]:.3f} | "
                 f"Recall {metrics[2]:.3f} | F1 {metrics[3]:.3f}", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, os.path.splitext(name)[0] + "_kitti_eval.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_kitti_eval():
    root = input("Path to KITTI training folder (contains image_2 and gt_image_2): ").strip().strip('"')
    img_dir = os.path.join(root, "image_2")
    gt_dir = os.path.join(root, "gt_image_2")
    if not (os.path.isdir(img_dir) and os.path.isdir(gt_dir)):
        print("Could not find image_2 and gt_image_2 inside that folder.")
        return

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(SUPPORTED))
    limit = input(f"How many images to evaluate? ({len(files)} available, Enter = all): ").strip()
    if limit.isdigit():
        files = files[:int(limit)]

    gt_index = build_gt_index(gt_dir)
    print(f"Found {len(files)} images and {sum(len(v) for v in gt_index.values())} mask files.")
    print("Example image names:", files[:3])
    print("Example mask names :", sorted(os.listdir(gt_dir))[:3])

    matplotlib.use("Agg")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, skipped = [], 0
    for i, name in enumerate(files, 1):
        gt_path, kind = find_kitti_gt(gt_index, name)
        if gt_path is None:
            skipped += 1
            continue
        image = load_image_rgb(os.path.join(img_dir, name))
        if image is None:
            skipped += 1
            continue
        gt = load_kitti_gt_road(gt_path, kind, image.shape[:2])
        if gt is None:
            skipped += 1
            continue

        pred = ml_pixel_wise_classification(traditional_cv_preprocess(image)) > 0
        m = seg_metrics(pred, gt)
        rows.append((name, kind, *m))
        print(f"[{i}/{len(files)}] {name}: IoU {m[0]:.3f}")
        if len(rows) <= 8:  # save example figures for the report
            save_kitti_figure(image, pred, gt, name, m)

    if not rows:
        print("No image/ground-truth pairs were found. Check the file names.")
        return

    arr = np.array([r[2:] for r in rows])
    print("\n" + "=" * 60)
    print(f"Evaluated {len(rows)} images ({skipped} skipped, no ground truth)")
    print(f"Mean IoU       : {arr[:, 0].mean():.3f}")
    print(f"Mean Precision : {arr[:, 1].mean():.3f}")
    print(f"Mean Recall    : {arr[:, 2].mean():.3f}")
    print(f"Mean F1        : {arr[:, 3].mean():.3f}")
    for prefix in ("um", "umm", "uu"):
        sub = [r[2] for r in rows if r[0].startswith(prefix + "_")]
        if sub:
            print(f"  {prefix:4s} subset IoU: {np.mean(sub):.3f} ({len(sub)} images)")
    print("=" * 60)

    path = os.path.join(RESULTS_DIR, "kitti_road_eval.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt_type", "iou", "precision", "recall", "f1"])
        for r in rows:
            writer.writerow([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]])
    print(f"Per-image results: {path}")
    print(f"Example figures saved in '{RESULTS_DIR}/'")


# =========================================================================
# 8c. TUSIMPLE U-NET EVALUATION (accuracy of the model you trained)
# =========================================================================
def _clahe_full(image_rgb):
    return traditional_cv_preprocess(image_rgb)


def evaluate_unet_on_tusimple(root, n_images, trained_on_these):
    frames_dir = os.path.join(root, "frames")
    masks_dir = os.path.join(root, "lane-masks")
    if not (os.path.isdir(frames_dir) and os.path.isdir(masks_dir)):
        print("Could not find 'frames' and 'lane-masks' inside that folder.")
        return
    if unet_model is None:
        print("The U-Net is not loaded, so there is nothing to evaluate.")
        return

    names = sorted(f for f in os.listdir(frames_dir)
                   if f.lower().endswith(SUPPORTED) and os.path.exists(os.path.join(masks_dir, f)))
    if not names:
        print("No frame/mask pairs with matching file names were found.")
        return
    rng = np.random.default_rng(0)  # fixed seed so results are repeatable
    if n_images and n_images < len(names):
        names = list(rng.choice(names, size=n_images, replace=False))
    print(f"Evaluating on {len(names)} frame/mask pairs...")

    variants = {"raw": lambda im: im}  # raw input won the earlier raw-vs-CLAHE comparison
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # ~3 px tolerance

    # accumulators[variant][thr] = [tp, fp, fn, sum_per_image_dice, tol_p_num, tol_p_den, tol_r_num, tol_r_den]
    acc = {v: {t: np.zeros(8) for t in thresholds} for v in variants}
    kept_examples = []

    batch_x = {v: [] for v in variants}
    batch_gt, batch_names = [], []

    def flush():
        if not batch_gt:
            return
        for v in variants:
            probs = unet_model(np.stack(batch_x[v]), training=False).numpy()[:, :, :, 0]
            for k, prob in enumerate(probs):
                gt = batch_gt[k]
                gt_tol = cv2.dilate(gt.astype(np.uint8), tol_kernel).astype(bool)
                for t in thresholds:
                    pred = prob > t
                    tp = np.logical_and(pred, gt).sum()
                    fp = np.logical_and(pred, ~gt).sum()
                    fn = np.logical_and(~pred, gt).sum()
                    dice_i = (2 * tp) / (2 * tp + fp + fn + 1e-9)
                    pred_tol = cv2.dilate(pred.astype(np.uint8), tol_kernel).astype(bool)
                    a = acc[v][t]
                    a += np.array([tp, fp, fn, dice_i,
                                   np.logical_and(pred, gt_tol).sum(), pred.sum(),
                                   np.logical_and(gt, pred_tol).sum(), gt.sum()])
                if len(kept_examples) < 6:
                    kept_examples.append((batch_names[k], batch_x[v][k], gt, prob))
        for v in variants:
            batch_x[v].clear()
        batch_gt.clear()
        batch_names.clear()

    for i, name in enumerate(names, 1):
        img = load_image_rgb_full(os.path.join(frames_dir, name))
        mask = cv2.imread(os.path.join(masks_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        gt = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_NEAREST) > 127
        for v, fn_prep in variants.items():
            prepared = fn_prep(img)
            small = cv2.resize(prepared, (512, 256), interpolation=cv2.INTER_AREA)
            batch_x[v].append(small.astype(np.float32) / 255.0)
        batch_gt.append(gt)
        batch_names.append(name)
        if len(batch_gt) == 16:
            flush()
            print(f"  {i}/{len(names)}")
    flush()

    # ---- report ----
    label = "TRAINING-SET score (optimistic)" if trained_on_these else "HELD-OUT score"
    print("\n" + "=" * 78)
    print(f"U-Net accuracy on {len(names)} frames  [{label}]")
    print("=" * 78)
    print(f"{'input':6s} {'thr':>4s} {'Dice':>7s} {'IoU':>7s} {'Prec':>7s} {'Recall':>7s} "
          f"{'meanDice':>9s} {'tolP':>7s} {'tolR':>7s}")
    best = None
    rows = []
    for v in variants:
        for t in thresholds:
            tp, fp, fn, dsum, tpn, tpd, trn, trd = acc[v][t]
            dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
            iou = tp / (tp + fp + fn + 1e-9)
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            mean_dice = dsum / len(names)
            tol_p = tpn / (tpd + 1e-9)
            tol_r = trn / (trd + 1e-9)
            print(f"{v:6s} {t:4.2f} {dice:7.3f} {iou:7.3f} {prec:7.3f} {rec:7.3f} "
                  f"{mean_dice:9.3f} {tol_p:7.3f} {tol_r:7.3f}")
            rows.append([v, t, dice, iou, prec, rec, mean_dice, tol_p, tol_r])
            if best is None or dice > best[2]:
                best = rows[-1]
    print("=" * 78)
    print(f"Best by Dice: input={best[0]}, threshold={best[1]}  (Dice {best[2]:.3f}, IoU {best[3]:.3f})")
    print("Dice/IoU are strict pixel overlap on very thin lines; tolP/tolR allow ~3 px of slack")
    print("and are usually the fairer numbers for line-like targets.")
    print(f"Set UNET_THRESHOLD = {best[1]} at the top of the script to use the best threshold.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "unet_tusimple_eval.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "threshold", "dice", "iou", "precision", "recall",
                         "mean_image_dice", "tolerant_precision", "tolerant_recall", "evaluated_on"])
        for r in rows:
            writer.writerow([r[0], r[1]] + [f"{x:.4f}" for x in r[2:]] + [label])
    print(f"Table saved: {path}")

    matplotlib.use("Agg")
    for name, x, gt, prob in kept_examples:
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
        axes[0].imshow(x)
        axes[0].set_title(f"Input ({name})")
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground truth lanes")
        axes[2].imshow(prob, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("U-Net probability")
        pred = prob > best[1]
        err = np.zeros((*gt.shape, 3), dtype=np.uint8)
        err[np.logical_and(pred, gt)] = (255, 255, 255)   # correct = white
        err[np.logical_and(pred, ~gt)] = (255, 0, 0)      # false alarm = red
        err[np.logical_and(~pred, gt)] = (0, 120, 255)    # missed = blue
        axes[3].imshow(err)
        axes[3].set_title("white=hit  red=false alarm  blue=missed")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, os.path.splitext(name)[0] + "_unet_eval.png"), dpi=100)
        plt.close(fig)
    print(f"Example figures saved in '{RESULTS_DIR}/'")


def load_image_rgb_full(path):
    """Loads a frame at full resolution (no width cap), as RGB."""
    bgr = cv2.imread(path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_unet_eval():
    root = input("Path to the TuSimple folder containing 'frames' and 'lane-masks'\n"
                 "(for example ...\\tusimple_preprocessed\\training): ").strip().strip('"')
    typed = input("How many frames to evaluate? (Enter = 150, 0 = all): ").strip()
    n = 150 if typed == "" else int(typed) if typed.isdigit() else 150
    n = None if n == 0 else n
    used = input("Was the model trained on these frames? (y/n): ").strip().lower() == "y"
    evaluate_unet_on_tusimple(root, n, used)


def run_single():
    image_path = select_single_image(TEST_DIR)
    if not image_path or not os.path.exists(image_path):
        print("No image selected.")
        return

    name = os.path.basename(image_path)
    image = load_image_rgb(image_path)
    if image is None:
        print("Could not read that image.")
        return

    truth = load_ground_truth().get(name)
    if truth is None:
        typed = input("True lane count for this image (press Enter to skip): ").strip()
        truth = int(typed) if typed.isdigit() else None

    print(f"\nProcessing: {image_path}")
    result = run_pipeline(image)
    print(f"Boundary lines found: {result['num_lines']}")
    print(f"Estimated lane count: {result['num_lanes']}")
    if truth is not None:
        print(f"True lane count: {truth} -> {'CORRECT' if truth == result['num_lanes'] else 'WRONG'}")

    out = save_result_figure(image, result, name, truth, show=True)
    print(f"Figure saved to {out}")


def prompt_missing_labels(files, truth_map):
    """
    For every image without a true lane count, show it and ask the user to type one.
    Enter = skip this image, q = stop asking. New labels are saved to labels.csv.
    """
    missing = [f for f in files if f not in truth_map]
    if not missing:
        return truth_map

    answer = input(f"\n{len(missing)} image(s) have no true lane count. "
                   f"Enter them now to measure accuracy? (y/n): ").strip().lower()
    if answer != "y":
        return truth_map

    print("For each image: close the preview window, then type the true number of lanes.")
    print("Press Enter to skip an image, or type q to stop.\n")
    new_rows = []
    for name in missing:
        image = load_image_rgb(os.path.join(TEST_DIR, name))
        if image is not None:
            plt.figure(figsize=(8, 5))
            plt.imshow(image)
            plt.title(name)
            plt.axis("off")
            plt.show()  # blocks until you close the window

        typed = input(f"True lanes for {name}: ").strip().lower()
        if typed == "q":
            break
        if typed.isdigit():
            truth_map[name] = int(typed)
            new_rows.append((name, int(typed)))

    if new_rows:
        file_exists = os.path.exists(LABELS_FILE)
        with open(LABELS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for name, value in new_rows:
                writer.writerow([name, value])
        print(f"\nSaved {len(new_rows)} label(s) to {LABELS_FILE} (reused on future runs).")
    return truth_map


def run_batch():
    files = list_test_images(TEST_DIR)
    if not files:
        print(f"No images found in '{TEST_DIR}/'.")
        return

    truth_map = load_ground_truth()
    truth_map = prompt_missing_labels(files, truth_map)
    matplotlib.use("Agg")  # no pop-up windows during processing
    rows = []
    for i, name in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {name}")
        image = load_image_rgb(os.path.join(TEST_DIR, name))
        if image is None:
            print("   could not read, skipped")
            continue
        result = run_pipeline(image)
        true = truth_map.get(name)
        save_result_figure(image, result, name, true)
        rows.append((name, result["num_lanes"], true))

    print_accuracy_report(rows)
    print(f"\nFigures saved in '{RESULTS_DIR}/'")
    print(f"Summary CSV: {save_summary_csv(rows)}")


if __name__ == "__main__":
    print("=" * 55)
    print("  1. Single image")
    print("  2. Batch: run every image in test_images/ + lane-count accuracy")
    print("  3. KITTI evaluation: road IoU against ground-truth masks")
    print("  4. U-Net evaluation: Dice/IoU on TuSimple frames + lane-masks")
    print("=" * 55)
    mode = input("Choose mode (1/2/3/4): ").strip()
    if mode == "4":
        run_unet_eval()
    elif mode == "3":
        run_kitti_eval()
    elif mode == "2":
        run_batch()
    else:
        run_single()
    print("Done!")
