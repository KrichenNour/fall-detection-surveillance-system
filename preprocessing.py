import cv2
import numpy as np
import os
import sys

# ---- Paths ----
data = sys.argv[1] if len(sys.argv) > 1 else "Fall"
Dataset_path = './Processed_Data/' + data
output_dir = f'./Processed_For_DL/' + data
output_dir_fg = os.path.join(output_dir, "fg")
output_dir_flow = os.path.join(output_dir, "flow")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_fg, exist_ok=True)
os.makedirs(output_dir_flow, exist_ok=True) 
for path in os.listdir(os.path.join(Dataset_path)):
    video_path = os.path.join(Dataset_path, path)
    cap = cv2.VideoCapture(video_path)

    # Check input
    ret, prev_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame:", path)
        continue

    h, w = prev_frame.shape[:2]

    # ---- Prepare background subtractor ----
    bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=40, detectShadows=False)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # ---- Containers for saving ----
    fg_masks = []        # list of foreground masks
    flows = []           # list of optical flow arrays

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ===== 1. Background Subtraction =====
        fgmask = bg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # ===== 2. Optical Flow =====
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # Save for deep learning
        fg_masks.append(fgmask.astype(np.uint8))         # (H, W)
        flows.append(flow.astype(np.float32))            # (H, W, 2)

        prev_gray = gray

    cap.release()

    # ---- Save arrays ----
    video_name = os.path.splitext(path)[0]
    np.save(os.path.join(output_dir_fg , f"{video_name}_fg.npy"), np.array(fg_masks))
    np.save(os.path.join(output_dir_flow , f"{video_name}_flow.npy"), np.array(flows))

    print("Saved processed data for video:", path)