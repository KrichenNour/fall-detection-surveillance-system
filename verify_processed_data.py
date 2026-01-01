import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

def verify_processed_data():
    """
    Verify the integrity of processed data in Processed_For_DL directory.
    Checks:
    1. All labels in CSV have corresponding files
    2. All files can be loaded without errors
    3. Files have correct shapes
    4. No corrupted numpy arrays
    """
    
    print("="*70)
    print("VERIFYING PROCESSED DATA INTEGRITY")
    print("="*70)
    
    # Paths
    labels_csv = "./Processed_For_DL/labels.csv"
    fg_fall_dir = "./Processed_For_DL/Fall/fg"
    flow_fall_dir = "./Processed_For_DL/Fall/flow"
    fg_no_fall_dir = "./Processed_For_DL/No_Fall/fg"
    flow_no_fall_dir = "./Processed_For_DL/No_Fall/flow"
    
    # Load labels
    print("\n[1] Loading labels from CSV...")
    if not os.path.exists(labels_csv):
        print(f"❌ ERROR: Labels file not found at {labels_csv}")
        return
    
    labels_df = pd.read_csv(labels_csv)
    print(f"✓ Loaded {len(labels_df)} labels from CSV")
    print(f"  - Fall samples (label=1): {(labels_df['label'] == 1).sum()}")
    print(f"  - No Fall samples (label=0): {(labels_df['label'] == 0).sum()}")
    
    # Check directories
    print("\n[2] Checking directories...")
    dirs_to_check = [
        (fg_fall_dir, "Fall/fg"),
        (flow_fall_dir, "Fall/flow"),
        (fg_no_fall_dir, "No_Fall/fg"),
        (flow_no_fall_dir, "No_Fall/flow")
    ]
    
    for dir_path, dir_name in dirs_to_check:
        if os.path.exists(dir_path):
            file_count = len(glob(os.path.join(dir_path, "*.npy")))
            print(f"✓ {dir_name:20s} - {file_count:4d} files")
        else:
            print(f"❌ {dir_name:20s} - Directory not found")
    
    # Check file existence and pairing
    print("\n[3] Checking file existence and pairing...")
    
    fall_stems = set()
    no_fall_stems = set()
    missing_files = []
    mismatched_pairs = []
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Checking files"):
        stem = row['filename']
        label = row['label']
        
        if label == 1:  # Fall
            fg_path = os.path.join(fg_fall_dir, f"{stem}_fg.npy")
            flow_path = os.path.join(flow_fall_dir, f"{stem}_flow.npy")
            category = "Fall"
        else:  # No Fall
            fg_path = os.path.join(fg_no_fall_dir, f"{stem}_fg.npy")
            flow_path = os.path.join(flow_no_fall_dir, f"{stem}_flow.npy")
            category = "No_Fall"
        
        fg_exists = os.path.exists(fg_path)
        flow_exists = os.path.exists(flow_path)
        
        if fg_exists and flow_exists:
            if label == 1:
                fall_stems.add(stem)
            else:
                no_fall_stems.add(stem)
        elif fg_exists or flow_exists:
            mismatched_pairs.append((stem, category, fg_exists, flow_exists))
        else:
            missing_files.append((stem, category))
    
    print(f"\n✓ Valid paired files:")
    print(f"  - Fall: {len(fall_stems)}")
    print(f"  - No_Fall: {len(no_fall_stems)}")
    print(f"  - Total: {len(fall_stems) + len(no_fall_stems)}")
    
    if missing_files:
        print(f"\n⚠ Missing files: {len(missing_files)}")
        print(f"  (Showing first 10)")
        for stem, category in missing_files[:10]:
            print(f"    - {stem} ({category})")
    
    if mismatched_pairs:
        print(f"\n⚠ Mismatched pairs (fg exists but flow doesn't or vice versa): {len(mismatched_pairs)}")
        print(f"  (Showing first 10)")
        for stem, category, fg_exists, flow_exists in mismatched_pairs[:10]:
            print(f"    - {stem} ({category}): fg={fg_exists}, flow={flow_exists}")
    
    # Check file integrity and shapes
    print("\n[4] Checking file integrity and shapes...")
    
    all_valid_stems = list(fall_stems) + list(no_fall_stems)
    corrupted_files = []
    shape_issues = []
    
    sample_count = min(100, len(all_valid_stems))  # Check first 100 or all if fewer
    print(f"  Sampling {sample_count} files for detailed checks...")
    
    for stem in tqdm(all_valid_stems[:sample_count], desc="Loading files"):
        # Determine label and paths
        if stem in fall_stems:
            fg_path = os.path.join(fg_fall_dir, f"{stem}_fg.npy")
            flow_path = os.path.join(flow_fall_dir, f"{stem}_flow.npy")
            label = 1
        else:
            fg_path = os.path.join(fg_no_fall_dir, f"{stem}_fg.npy")
            flow_path = os.path.join(flow_no_fall_dir, f"{stem}_flow.npy")
            label = 0
        
        try:
            # Load foreground
            fg = np.load(fg_path, allow_pickle=False)
            if fg.ndim != 3:
                shape_issues.append((stem, "fg", f"Expected 3D, got {fg.ndim}D", fg.shape))
            
            # Load flow
            flow = np.load(flow_path, allow_pickle=False)
            if flow.ndim != 4:
                shape_issues.append((stem, "flow", f"Expected 4D, got {flow.ndim}D", flow.shape))
            
            # Check if fg and flow have same sequence length
            if fg.shape[0] != flow.shape[0]:
                shape_issues.append((stem, "both", f"Sequence length mismatch", f"fg:{fg.shape[0]} vs flow:{flow.shape[0]}"))
            
        except Exception as e:
            corrupted_files.append((stem, str(e)))
    
    if corrupted_files:
        print(f"\n❌ Corrupted files found: {len(corrupted_files)}")
        for stem, error in corrupted_files[:10]:
            print(f"    - {stem}: {error}")
    else:
        print(f"\n✓ No corrupted files in sample")
    
    if shape_issues:
        print(f"\n⚠ Shape issues found: {len(shape_issues)}")
        for stem, file_type, issue, details in shape_issues[:10]:
            print(f"    - {stem} ({file_type}): {issue} - {details}")
    else:
        print(f"\n✓ All sampled files have correct shapes")
    
    # Check for orphaned files (files not in CSV)
    print("\n[5] Checking for orphaned files (not in CSV)...")
    
    csv_stems = set(labels_df['filename'].values)
    
    orphaned = []
    for dir_path, dir_name in [(fg_fall_dir, "Fall/fg"), (flow_fall_dir, "Fall/flow"),
                                 (fg_no_fall_dir, "No_Fall/fg"), (flow_no_fall_dir, "No_Fall/flow")]:
        if os.path.exists(dir_path):
            for file in glob(os.path.join(dir_path, "*.npy")):
                basename = os.path.basename(file)
                stem = basename.replace("_fg.npy", "").replace("_flow.npy", "")
                if stem not in csv_stems:
                    orphaned.append((stem, dir_name, basename))
    
    if orphaned:
        print(f"⚠ Found {len(orphaned)} orphaned files (not in CSV)")
        print(f"  (Showing first 10)")
        for stem, dir_name, filename in orphaned[:10]:
            print(f"    - {filename} in {dir_name}")
    else:
        print(f"✓ No orphaned files found")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total labels in CSV:        {len(labels_df)}")
    print(f"Valid paired files:         {len(fall_stems) + len(no_fall_stems)}")
    print(f"  - Fall:                   {len(fall_stems)}")
    print(f"  - No_Fall:                {len(no_fall_stems)}")
    print(f"Missing files:              {len(missing_files)}")
    print(f"Mismatched pairs:           {len(mismatched_pairs)}")
    print(f"Corrupted files (sampled):  {len(corrupted_files)}")
    print(f"Shape issues (sampled):     {len(shape_issues)}")
    print(f"Orphaned files:             {len(orphaned)}")
    
    coverage = (len(fall_stems) + len(no_fall_stems)) / len(labels_df) * 100
    print(f"\nData coverage: {coverage:.1f}%")
    
    if corrupted_files or len(mismatched_pairs) > 0:
        print("\n⚠ WARNING: Issues detected. Review above details.")
    elif coverage < 50:
        print(f"\n⚠ WARNING: Low coverage ({coverage:.1f}%). Most videos need processing.")
    else:
        print("\n✓ Data integrity looks good!")
    
    print("="*70)

if __name__ == "__main__":
    verify_processed_data()
