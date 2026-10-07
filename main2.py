import os
import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

try:
    from moviepy.editor import VideoFileClip

    USE_LEGACY_MOVIEPY = True
except ImportError:
    from moviepy import VideoFileClip

    USE_LEGACY_MOVIEPY = False

# Enable multi-core CPU threading for 12th Gen Intel
torch.set_num_threads(os.cpu_count() or 4)

# =========================================================================
# 1. LOAD PRETRAINED SEGFORMER
# =========================================================================
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

print("Loading SegFormer model...")
feature_extractor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()
print("Model initialized for video pipeline.\n")


# =========================================================================
# 2. VIDEO FILE PICKER
# =========================================================================
def select_input_video(directory="test_videos"):
    os.makedirs(directory, exist_ok=True)
    supported_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported_extensions)]

    print("=" * 55)
    print(f"       SELECT A VIDEO FROM '{directory}/'")
    print("=" * 55)
    for i, file_name in enumerate(files, 1):
        print(f" {i}. {file_name}")
    print(" 0. Browse any video from computer (Open File Dialog)")
    print("=" * 55)

    choice = input(f"Enter choice (0-{len(files)}): ").strip()
    if choice == "0":
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Road Video",
            filetypes=[("Video Files", "*.mp4;*.mov;*.avi;*.mkv"), ("All Files", "*.*")]
        )
        root.destroy()
        return path
    elif choice.isdigit() and 1 <= int(choice) <= len(files):
        return os.path.join(directory, files[int(choice) - 1])
    elif len(files) > 0:
        return os.path.join(directory, files[0])
    return None


# =========================================================================
# 3. HIGH-STABILITY EGO-LANE & MULTI-LANE DETECTOR
# =========================================================================
class StableLaneDetector:
    def __init__(self, keyframe_interval=4, alpha=0.30):
        self.interval = keyframe_interval
        self.alpha = alpha
        self.frame_count = 0
        self.cached_road_mask = None

        # State memory for smooth temporal transitions
        self.left_fit_prev = None
        self.right_fit_prev = None

        self.kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        self.kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def process_frame(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        self.frame_count += 1

        # -----------------------------------------------------------------
        # STEP 1: ML Road Surface (Keyframe Cadence)
        # -----------------------------------------------------------------
        if self.cached_road_mask is None or (self.frame_count % self.interval == 0):
            proc_w, proc_h = 448, 256
            small_frame = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

            inputs = feature_extractor(images=small_frame, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            pred_seg = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            small_mask = (pred_seg == 0).astype(np.uint8) * 255
            self.cached_road_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        road_mask = self.cached_road_mask.copy()

        # Remove car hood & interior rearview mirror / rosaries:
        # Zero out bottom 10% (dashboard) and top 12% of the frame
        road_mask[int(h * 0.90):, :] = 0
        road_mask[:int(h * 0.12), :] = 0

        # -----------------------------------------------------------------
        # STEP 2: Top-Hat Glare Filter & Lane Extraction
        # -----------------------------------------------------------------
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel_tophat)

        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([12, 60, 80]), np.array([38, 255, 255]))

        _, tophat_thresh = cv2.threshold(tophat, 16, 255, cv2.THRESH_BINARY)
        combined_stripes = cv2.bitwise_or(tophat_thresh, yellow_mask)
        lane_pixels = cv2.bitwise_and(combined_stripes, road_mask)
        clean_lanes = cv2.morphologyEx(lane_pixels, cv2.MORPH_OPEN, self.kernel_clean)

        # -----------------------------------------------------------------
        # STEP 3: Ego-Lane Segmentation (Left & Right Boundaries)
        # -----------------------------------------------------------------
        road_y_coords = np.nonzero(road_mask)[0]
        y_horizon = int(np.percentile(road_y_coords, 3)) if len(road_y_coords) > 0 else int(h * 0.55)
        y_horizon = max(int(h * 0.48), min(y_horizon, int(h * 0.65)))
        y_bottom = int(h * 0.90)

        lines = cv2.HoughLinesP(
            clean_lanes, rho=1, theta=np.pi / 180, threshold=15,
            minLineLength=20, maxLineGap=40
        )

        left_x, left_y = [], []
        right_x, right_y = [], []
        midpoint_x = w // 2

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(-1)
                if y1 == y2:
                    continue
                if y1 < y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1

                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else -999.0

                # Left Lane: negative slope (in image coordinates) and located left of center
                if -2.5 < slope < -0.4 and (x1 < midpoint_x + 60 and x2 < midpoint_x + 60):
                    left_x.extend([x1, x2])
                    left_y.extend([y1, y2])

                # Right Lane: positive slope and located right of center
                elif 0.4 < slope < 2.5 and (x1 > midpoint_x - 60 and x2 > midpoint_x - 60):
                    right_x.extend([x1, x2])
                    right_y.extend([y1, y2])

        # Temporal fit for Left Lane
        current_left_fit = None
        if len(left_x) >= 4:
            current_left_fit = np.polyfit(left_y, left_x, 1)
            if self.left_fit_prev is not None:
                current_left_fit = self.alpha * current_left_fit + (1 - self.alpha) * self.left_fit_prev
            self.left_fit_prev = current_left_fit
        else:
            current_left_fit = self.left_fit_prev

        # Temporal fit for Right Lane
        current_right_fit = None
        if len(right_x) >= 4:
            current_right_fit = np.polyfit(right_y, right_x, 1)
            if self.right_fit_prev is not None:
                current_right_fit = self.alpha * current_right_fit + (1 - self.alpha) * self.right_fit_prev
            self.right_fit_prev = current_right_fit
        else:
            current_right_fit = self.right_fit_prev

        # -----------------------------------------------------------------
        # STEP 4: Render Visuals
        # -----------------------------------------------------------------
        overlay = frame_rgb.copy()
        # Drivable road surface
        overlay[road_mask > 0] = overlay[road_mask > 0] * 0.78 + np.array([0, 180, 0], dtype=np.uint8) * 0.22

        line_canvas = np.zeros_like(frame_rgb)

        # Draw verified left lane
        if current_left_fit is not None:
            x_bot_l = int(current_left_fit[0] * y_bottom + current_left_fit[1])
            x_top_l = int(current_left_fit[0] * y_horizon + current_left_fit[1])
            cv2.line(line_canvas, (x_bot_l, y_bottom), (x_top_l, y_horizon), (255, 0, 0), 6)

        # Draw verified right lane
        if current_right_fit is not None:
            x_bot_r = int(current_right_fit[0] * y_bottom + current_right_fit[1])
            x_top_r = int(current_right_fit[0] * y_horizon + current_right_fit[1])
            cv2.line(line_canvas, (x_bot_r, y_bottom), (x_top_r, y_horizon), (255, 0, 0), 6)

        overlay[line_canvas[:, :, 0] > 0] = [255, 0, 0]
        return overlay


# =========================================================================
# 4. ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    video_path = select_input_video("test_videos")
    if not video_path or not os.path.exists(video_path):
        print("No valid video selected.")
        exit(0)

    output_dir = "test_videos_output"
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"stable_{os.path.basename(video_path)}"
    output_path = os.path.join(output_dir, out_name)

    print(f"\nProcessing Video: {video_path}")
    print(f"Output: {output_path}")

    detector = StableLaneDetector(keyframe_interval=4, alpha=0.30)
    clip = VideoFileClip(video_path)

    if USE_LEGACY_MOVIEPY:
        processed_clip = clip.fl_image(detector.process_frame)
    else:
        processed_clip = clip.image_transform(detector.process_frame)

    processed_clip.write_videofile(output_path, audio=False, preset="ultrafast", threads=4)
    print(f"\nFinished! Stable video saved to: {output_path}")