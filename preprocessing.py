import cv2
import os
import numpy as np
import subprocess
from tqdm import tqdm
import json

# ============================================
# CONFIGURATION
# ============================================

CONFIG = {
    # Input directories
    "input_dirs": {
        #"fall": r"D:\download\archive\Fall\Raw_Video",
        "no_fall": r"D:\download\archive\No_Fall\Raw_Video"
    },
    
    # Output directory
    "output_dir": r"D:\download\Preprocessed_Videos",
    
    # Video processing parameters
    "resize_shape": (224, 224),  # Resize frames (None to keep original)
    "target_fps": 30,             # Standardize FPS (None to keep original)
    
    # Background subtraction parameters
    "bg_method": "MOG2",          # Options: "MOG2", "KNN"
    "history": 500,               # Number of frames for background model
    "var_threshold": 16,          # Threshold for MOG2
    "detect_shadows": False,      # Detect shadows (slower but more accurate)
    
    # Morphological operations (to clean foreground mask)
    "morphology": {
        "enabled": True,
        "kernel_size": 5,         # Size of morphological kernel
        "opening_iterations": 2,  # Remove noise
        "closing_iterations": 2   # Fill holes
    },
    
    # Optical flow parameters
    "optical_flow": {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2
    }
}

# ============================================
# BACKGROUND SUBTRACTION METHODS
# ============================================

def create_background_subtractor(method="MOG2", history=500, var_threshold=16, detect_shadows=False):
    """
    Create background subtractor object
    
    Methods:
    - MOG2: Gaussian Mixture-based (best for most cases)
    - KNN: K-Nearest Neighbors (faster, good for simple backgrounds)
    """
    if method == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
    elif method == "KNN":
        return cv2.createBackgroundSubtractorKNN(
            history=history,
            dist2Threshold=400.0,
            detectShadows=detect_shadows
        )
    else:
        raise ValueError(f"Unknown background subtraction method: {method}")

# ============================================
# MORPHOLOGICAL OPERATIONS
# ============================================

def clean_foreground_mask(mask, kernel_size=5, opening_iter=2, closing_iter=2):
    """
    Clean foreground mask using morphological operations
    
    Steps:
    1. Opening: Remove small noise (erosion followed by dilation)
    2. Closing: Fill small holes (dilation followed by erosion)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=opening_iter)
    
    # Closing: fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iter)
    
    return mask

# ============================================
# FPS STANDARDIZATION
# ============================================

def standardize_fps(video_path, temp_path, target_fps):
    """Use FFmpeg to convert video to target FPS"""
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-filter:v', f'fps=fps={target_fps}',
        '-c:v', 'libx264', '-preset', 'fast',
        temp_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Warning: FFmpeg failed for {video_path}")
        return video_path
    return temp_path

# ============================================
# MAIN PROCESSING FUNCTION
# ============================================

def process_video_with_bg_subtraction(video_path, config):
    """
    Process single video with background subtraction and optical flow
    
    Returns:
        dict with:
        - optical_flow: numpy array of shape (num_frames-1, H, W, 2)
        - foreground_masks: numpy array of shape (num_frames, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video info: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
    
    # Initialize background subtractor
    bg_subtractor = create_background_subtractor(
        method=config["bg_method"],
        history=config["history"],
        var_threshold=config["var_threshold"],
        detect_shadows=config["detect_shadows"]
    )
    
    # Storage for results
    optical_flows = []
    foreground_masks = []
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    if config["resize_shape"]:
        prev_frame = cv2.resize(prev_frame, config["resize_shape"])
    
    # Apply background subtraction to first frame
    prev_fg_mask = bg_subtractor.apply(prev_frame)
    
    # Clean mask
    if config["morphology"]["enabled"]:
        prev_fg_mask = clean_foreground_mask(
            prev_fg_mask,
            kernel_size=config["morphology"]["kernel_size"],
            opening_iter=config["morphology"]["opening_iterations"],
            closing_iter=config["morphology"]["closing_iterations"]
        )
    
    foreground_masks.append(prev_fg_mask)
    
    # Convert to grayscale for optical flow
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply mask to grayscale (focus optical flow on foreground)
    prev_gray_masked = cv2.bitwise_and(prev_gray, prev_gray, mask=prev_fg_mask)
    
    pbar = tqdm(total=frame_count-1, desc="  Processing frames", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if config["resize_shape"]:
            frame = cv2.resize(frame, config["resize_shape"])
        
        # ===== STEP 1: Background Subtraction =====
        fg_mask = bg_subtractor.apply(frame)
        
        # Clean mask
        if config["morphology"]["enabled"]:
            fg_mask = clean_foreground_mask(
                fg_mask,
                kernel_size=config["morphology"]["kernel_size"],
                opening_iter=config["morphology"]["opening_iterations"],
                closing_iter=config["morphology"]["closing_iterations"]
            )
        
        foreground_masks.append(fg_mask)
        
        # ===== STEP 2: Optical Flow on Foreground =====
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=fg_mask)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray_masked, gray_masked, None,
            pyr_scale=config["optical_flow"]["pyr_scale"],
            levels=config["optical_flow"]["levels"],
            winsize=config["optical_flow"]["winsize"],
            iterations=config["optical_flow"]["iterations"],
            poly_n=config["optical_flow"]["poly_n"],
            poly_sigma=config["optical_flow"]["poly_sigma"],
            flags=0
        )
        
        optical_flows.append(flow)
        
        # Update for next iteration
        prev_gray_masked = gray_masked
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if len(optical_flows) == 0:
        return None
    
    return {
        'optical_flow': np.array(optical_flows),      # (T-1, H, W, 2)
        'foreground_masks': np.array(foreground_masks) # (T, H, W)
    }

# ============================================
# BATCH PROCESSING
# ============================================

def process_videos(label, folder_path, output_dir, config):
    """Process all videos in a folder"""
    
    # Create output directories
    out_label_dir = os.path.join(output_dir, label)
    out_flow_dir = os.path.join(out_label_dir, "optical_flow")
    out_mask_dir = os.path.join(out_label_dir, "foreground_masks")
    
    os.makedirs(out_flow_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    # Get video files
    video_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    print(f"\n{'='*60}")
    print(f"Processing {len(video_files)} videos from: {label}")
    print(f"{'='*60}")
    
    stats = {
        'total': len(video_files),
        'successful': 0,
        'failed': 0,
        'failed_videos': []
    }
    
    for vid_name in tqdm(video_files, desc=f"Processing {label}"):
        video_path = os.path.join(folder_path, vid_name)
        base_name = os.path.splitext(vid_name)[0]
        
        try:
            # Standardize FPS if needed
            temp_path = video_path
            if config["target_fps"]:
                temp_path = os.path.join(folder_path, f"temp_{vid_name}")
                temp_path = standardize_fps(video_path, temp_path, config["target_fps"])
            
            # Process video
            result = process_video_with_bg_subtraction(temp_path, config)
            
            if result is not None:
                # Save optical flow
                flow_path = os.path.join(out_flow_dir, f"{base_name}_flow.npy")
                np.save(flow_path, result['optical_flow'])
                
                # Save foreground masks
                mask_path = os.path.join(out_mask_dir, f"{base_name}_mask.npy")
                np.save(mask_path, result['foreground_masks'])
                
                stats['successful'] += 1
                
            else:
                stats['failed'] += 1
                stats['failed_videos'].append(vid_name)
                print(f"  ❌ Failed: {vid_name}")
            
            # Clean up temporary file
            if temp_path != video_path and os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            stats['failed'] += 1
            stats['failed_videos'].append(vid_name)
            print(f"  ❌ Error processing {vid_name}: {str(e)}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Statistics for {label}:")
    print(f"  ✓ Successful: {stats['successful']}/{stats['total']}")
    print(f"  ✗ Failed: {stats['failed']}/{stats['total']}")
    if stats['failed_videos']:
        print(f"  Failed videos: {stats['failed_videos'][:5]}")
        if len(stats['failed_videos']) > 5:
            print(f"    ... and {len(stats['failed_videos'])-5} more")
    print(f"{'='*60}")
    
    return stats

# ============================================
# HELPER FUNCTIONS
# ============================================

def save_statistics(stats, output_dir):
    """Save processing statistics to JSON"""
    stats_path = os.path.join(output_dir, "preprocessing_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Statistics saved to: {stats_path}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("FALL DETECTION - VIDEO PREPROCESSING")
    print("WITH BACKGROUND SUBTRACTION (LIGHTWEIGHT)")
    print("="*60)
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(CONFIG["output_dir"], "preprocessing_config.json")
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")
    
    # Process all videos
    all_stats = {}
    
    for label, folder in CONFIG["input_dirs"].items():
        if not os.path.exists(folder):
            print(f"\n⚠️  Warning: Folder not found: {folder}")
            continue
        
        stats = process_videos(label, folder, CONFIG["output_dir"], CONFIG)
        all_stats[label] = stats
    
    # Save overall statistics
    save_statistics(all_stats, CONFIG["output_dir"])
    
    # Print final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print(f"\nDirectory structure:")
    print(f"  {CONFIG['output_dir']}/")
    for label in CONFIG["input_dirs"].keys():
        print(f"  ├── {label}/")
        print(f"  │   ├── optical_flow/     (*.npy files)")
        print(f"  │   └── foreground_masks/ (*.npy files)")
    
    print(f"\n✓ Total videos processed: {sum(s['successful'] for s in all_stats.values())}")
    print(f"✓ Total videos failed: {sum(s['failed'] for s in all_stats.values())}")
    print("\nReady for model training!")