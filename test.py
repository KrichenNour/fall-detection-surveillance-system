"""
Two-Stream Transformer Fall Detection - Multi-Scenario Video Testing
Analyzes long videos with multiple fall scenarios using sliding window
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================================================
# MODEL DEFINITION
# ==============================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, patch_size, d_model, img_size):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x): 
        x = self.proj(x)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x

def transformer_encoder_layer(d_model, nhead, mlp_ratio, depth, dropout=0.3):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead,
        dim_feedforward=int(d_model * mlp_ratio), 
        dropout=dropout, activation='gelu'
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=depth)

class TwoStreamTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=64, fg_in_ch=1, flow_in_ch=2, 
                 d_model=64, depth=1, num_heads=4, mlp_ratio=2, dropout=0.3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.d_model = d_model
        self.seq_tokens = 63 * self.num_patches_per_frame

        self.fg_patch_embed = PatchEmbedding(fg_in_ch, patch_size, d_model, img_size)
        self.flow_patch_embed = PatchEmbedding(flow_in_ch, patch_size, d_model, img_size)

        self.cls_fg_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_flow_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_embed_fg = nn.Parameter(torch.zeros(1, 1 + self.seq_tokens, d_model))
        self.pos_embed_flow = nn.Parameter(torch.zeros(1, 1 + self.seq_tokens, d_model))

        self.encoder_fg = transformer_encoder_layer(d_model, num_heads, mlp_ratio, depth, dropout)
        self.encoder_flow = transformer_encoder_layer(d_model, num_heads, mlp_ratio, depth, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(d_model, 1)
        )

        nn.init.trunc_normal_(self.pos_embed_fg, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_flow, std=0.02)
        nn.init.trunc_normal_(self.cls_fg_token, std=0.02)
        nn.init.trunc_normal_(self.cls_flow_token, std=0.02)
    
    def forward(self, fg_x, flow_x):
        B, T, C1, H, W = fg_x.shape
        B, T, C2, H, W = flow_x.shape
        
        fg_x = fg_x.view(B*T, C1, H, W)
        flow_x = flow_x.view(B*T, C2, H, W)

        fg_x = self.fg_patch_embed(fg_x)
        flow_x = self.flow_patch_embed(flow_x)
        
        P = fg_x.size(1)
        fg_tokens = fg_x.reshape(B, T*P, self.d_model)
        flow_tokens = flow_x.reshape(B, T*P, self.d_model)

        cls_fg = self.cls_fg_token.expand(B, -1, -1)
        cls_flow = self.cls_flow_token.expand(B, -1, -1)
        fg_tokens = torch.cat((cls_fg, fg_tokens), dim=1)
        flow_tokens = torch.cat((cls_flow, flow_tokens), dim=1)

        fg_tokens = fg_tokens + self.pos_embed_fg
        flow_tokens = flow_tokens + self.pos_embed_flow

        fg_out = self.encoder_fg(fg_tokens.transpose(0, 1)).transpose(0, 1)
        flow_out = self.encoder_flow(flow_tokens.transpose(0, 1)).transpose(0, 1)

        cls_fg_out = fg_out[:, 0, :]
        cls_flow_out = flow_out[:, 0, :]
        fused = torch.cat((cls_fg_out, cls_flow_out), dim=1)

        logits = self.classifier(fused)
        return logits.squeeze(1)

# ==============================================================
# VIDEO PREPROCESSING - FULL VIDEO
# ==============================================================

def preprocess_full_video(video_path, img_size=224):
    """
    Preprocess entire video and extract all frames
    """
    
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"  Duration: {duration:.2f}s | FPS: {fps:.2f} | Frames: {frame_count}")
    
    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    
    # Read all frames
    all_frames = []
    all_fg_masks = []
    
    print("  Reading all frames...")
    pbar = tqdm(total=frame_count, desc="  Progress")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (img_size, img_size))
        all_frames.append(frame)
        
        # Background subtraction
        fg_mask = bg_subtractor.apply(frame)
        fg_mask = (fg_mask / 255.0).astype(np.float32)
        all_fg_masks.append(fg_mask)
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"  Read {len(all_frames)} frames")
    
    # Compute optical flow for all consecutive frames
    print("  Computing optical flow...")
    optical_flows = []
    
    for i in tqdm(range(len(all_frames)-1), desc="  Progress"):
        prev_frame = all_frames[i]
        curr_frame = all_frames[i+1]
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply masks
        prev_mask = (all_fg_masks[i] * 255).astype(np.uint8)
        curr_mask = (all_fg_masks[i+1] * 255).astype(np.uint8)
        
        prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=prev_mask)
        curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=curr_mask)
        
        # Calculate flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        optical_flows.append(flow)
    
    # Add last flow (duplicate)
    optical_flows.append(optical_flows[-1])
    
    print(f"  Preprocessing complete\n")
    
    return all_fg_masks, optical_flows, fps

# ==============================================================
# SLIDING WINDOW PREDICTION
# ==============================================================

def analyze_video_sliding_window(video_path, model, device, window_size=63, stride=20):
    """
    Analyze video using sliding window approach
    
    Args:
        video_path: Path to video
        model: Trained model
        device: cuda or cpu
        window_size: Number of frames per window (63)
        stride: Step size between windows (20 frames = ~0.67s at 30fps)
    """
    
    # Preprocess entire video
    fg_masks, flows, fps = preprocess_full_video(video_path)
    
    total_frames = len(fg_masks)
    
    if total_frames < window_size:
        print(f"Warning: Video has only {total_frames} frames, need at least {window_size}")
        print("Padding with last frame...")
        padding = window_size - total_frames
        fg_masks.extend([fg_masks[-1]] * padding)
        flows.extend([flows[-1]] * padding)
        total_frames = window_size
    
    # Calculate number of windows
    num_windows = (total_frames - window_size) // stride + 1
    
    print(f"Analyzing video with sliding window:")
    print(f"  Window size: {window_size} frames")
    print(f"  Stride: {stride} frames (~{stride/fps:.2f}s)")
    print(f"  Number of windows: {num_windows}\n")
    
    predictions = []
    probabilities = []
    timestamps = []
    
    print("Making predictions...")
    for i in tqdm(range(num_windows), desc="  Windows"):
        start_idx = i * stride
        end_idx = start_idx + window_size
        
        # Extract window
        window_fg = fg_masks[start_idx:end_idx]
        window_flow = flows[start_idx:end_idx]
        
        # Convert to arrays
        fg_array = np.stack(window_fg, axis=0)
        fg_array = np.expand_dims(fg_array, axis=1)  # (63, 1, 224, 224)
        
        flow_array = np.stack(window_flow, axis=0)
        flow_array = np.transpose(flow_array, (0, 3, 1, 2))  # (63, 2, 224, 224)
        
        # Normalize flow
        flow_clip = 20.0
        flow_array = np.clip(flow_array, -flow_clip, flow_clip) / flow_clip
        
        # Convert to tensors
        fg_tensor = torch.from_numpy(fg_array).float().unsqueeze(0).to(device)
        flow_tensor = torch.from_numpy(flow_array).float().unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            logits = model(fg_tensor, flow_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability >= 0.5 else 0
        
        predictions.append(prediction)
        probabilities.append(probability)
        
        # Calculate timestamp (middle of window)
        timestamp = (start_idx + window_size // 2) / fps
        timestamps.append(timestamp)
    
    return predictions, probabilities, timestamps, fps

# ==============================================================
# RESULTS ANALYSIS
# ==============================================================

def analyze_results(predictions, probabilities, timestamps):
    """Analyze and display results"""
    
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    fall_count = sum(predictions)
    no_fall_count = len(predictions) - fall_count
    
    print(f"\nTotal windows analyzed: {len(predictions)}")
    print(f"  Fall detections: {fall_count} ({fall_count/len(predictions)*100:.1f}%)")
    print(f"  No-Fall detections: {no_fall_count} ({no_fall_count/len(predictions)*100:.1f}%)")
    
    # Find fall events (consecutive fall predictions)
    fall_events = []
    in_fall = False
    fall_start = 0
    
    for i, pred in enumerate(predictions):
        if pred == 1 and not in_fall:
            in_fall = True
            fall_start = i
        elif pred == 0 and in_fall:
            in_fall = False
            fall_events.append((fall_start, i-1))
    
    if in_fall:
        fall_events.append((fall_start, len(predictions)-1))
    
    print(f"\nDetected {len(fall_events)} fall event(s):")
    for idx, (start, end) in enumerate(fall_events, 1):
        start_time = timestamps[start]
        end_time = timestamps[end]
        duration = end_time - start_time
        max_conf = max(probabilities[start:end+1]) * 100
        print(f"  Event {idx}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s, max confidence: {max_conf:.1f}%)")
    
    print("="*70 + "\n")
    
    return fall_events

# ==============================================================
# VISUALIZATION
# ==============================================================

def visualize_timeline(predictions, probabilities, timestamps, video_path, fall_events):
    """Create timeline visualization"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Predictions over time
    colors = ['red' if p == 1 else 'green' for p in predictions]
    ax1.bar(timestamps, [1]*len(predictions), color=colors, alpha=0.6, width=timestamps[1]-timestamps[0])
    ax1.set_ylabel('Prediction', fontsize=12, fontweight='bold')
    ax1.set_title(f'Fall Detection Timeline - {os.path.basename(video_path)}', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.5])
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    
    # Mark fall events
    for idx, (start, end) in enumerate(fall_events, 1):
        mid = (timestamps[start] + timestamps[end]) / 2
        ax1.text(mid, 1.2, f'Fall Event {idx}', ha='center', fontweight='bold', color='red')
    
    ax1.legend(['Red = Fall', 'Green = No Fall'], loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Confidence over time
    ax2.plot(timestamps, probabilities, linewidth=2, color='blue')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.fill_between(timestamps, probabilities, alpha=0.3)
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fall Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Highlight fall events
    for start, end in fall_events:
        ax2.axvspan(timestamps[start], timestamps[end], alpha=0.2, color='red')
    
    plt.tight_layout()
    
    output_name = f"timeline_{os.path.splitext(os.path.basename(video_path))[0]}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"Timeline saved to: {output_name}")
    plt.close()

# ==============================================================
# MAIN PREDICTION FUNCTION
# ==============================================================

def predict(video_path, model_path="best_two_stream_transformer.pth", 
            window_size=63, stride=20):
    """
    Analyze video for fall detection
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model
        window_size: Frames per window (default: 63)
        stride: Frames between windows (default: 20)
    """
    
    print("\n" + "="*70)
    print("TWO-STREAM TRANSFORMER - FALL DETECTION")
    print("MULTI-SCENARIO VIDEO ANALYSIS")
    print("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = TwoStreamTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded\n")
    
    # Analyze video
    predictions, probabilities, timestamps, fps = analyze_video_sliding_window(
        video_path, model, device, window_size, stride
    )
    
    # Analyze results
    fall_events = analyze_results(predictions, probabilities, timestamps)
    
    # Visualize
    visualize_timeline(predictions, probabilities, timestamps, video_path, fall_events)
    
    return predictions, probabilities, timestamps, fall_events

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    
    print("="*70)
    print("SCRIPT STARTED")
    print("="*70)
    
    # ===== CONFIGURATION =====
    VIDEO_PATH = "test_video.mp4"
    MODEL_PATH = "best_two_stream_transformer.pth"
    
    # Sliding window parameters
    WINDOW_SIZE = 63    # frames per window (fixed for model)
    STRIDE = 20         # frames between windows (20 frames = ~0.67s at 30fps)
    
    print(f"\nConfiguration:")
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  Stride: {STRIDE}")
    
    # Check if files exist
    print(f"\nChecking files...")
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"Files in current directory:")
        for f in os.listdir():
            print(f"  - {f}")
        exit(1)
    else:
        print(f"  Video found: {VIDEO_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"Files in current directory:")
        for f in os.listdir():
            print(f"  - {f}")
        exit(1)
    else:
        print(f"  Model found: {MODEL_PATH}")
    
    # ===== RUN PREDICTION =====
    try:
        print("\n" + "="*70)
        print("STARTING PREDICTION")
        print("="*70 + "\n")
        
        predictions, probabilities, timestamps, fall_events = predict(
            VIDEO_PATH, 
            MODEL_PATH,
            window_size=WINDOW_SIZE,
            stride=STRIDE
        )
        
        print("\n" + "="*70)
        print("TESTING COMPLETE!")
        print("="*70)
        print(f"\nTotal windows analyzed: {len(predictions)}")
        print(f"Fall events detected: {len(fall_events)}")
        print(f"\nResults saved!")
        
    except FileNotFoundError as e:
        print(f"\nERROR: File not found - {e}")
        print("\nPlease make sure:")
        print(f"  1. Video file exists: {VIDEO_PATH}")
        print(f"  2. Model file exists: {MODEL_PATH}")
        import traceback
        traceback.print_exc()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()