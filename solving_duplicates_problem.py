import pandas as pd
import os
import shutil

def fix_duplicate_filenames():
    labels_csv = "./Processed_For_DL/labels.csv"
    
    # Directories
    fg_fall_dir = "./Processed_For_DL/Fall/fg"
    flow_fall_dir = "./Processed_For_DL/Fall/flow"
    fg_no_fall_dir = "./Processed_For_DL/No_Fall/fg"
    flow_no_fall_dir = "./Processed_For_DL/No_Fall/flow"
    
    # Read CSV
    df = pd.read_csv(labels_csv)
    print(f"Total rows in CSV: {len(df)}")
    
    # Find duplicates
    duplicates = df[df.duplicated(subset=['filename'], keep=False)].sort_values('filename')
    
    if len(duplicates) == 0:
        print("✅ No duplicates found!")
        return
    
    print(f"\n⚠️  Found {len(duplicates)} duplicate entries!")
    
    # Group duplicates by filename
    duplicate_groups = duplicates.groupby('filename')
    
    # Track renamed files
    renamed_files = []
    updated_rows = []
    
    for filename, group in duplicate_groups:
        print(f"\n📝 Processing duplicate: {filename}")
        
        for idx, row in group.iterrows():
            label = row['label']
            old_stem = filename
            
            # Determine new name based on label
            if label == 1:  # Fall
                new_stem = f"{old_stem}_fall"
                fg_dir = fg_fall_dir
                flow_dir = flow_fall_dir
                label_name = "Fall"
            else:  # No Fall
                new_stem = f"{old_stem}_no_fall"
                fg_dir = fg_no_fall_dir
                flow_dir = flow_no_fall_dir
                label_name = "No_Fall"
            
            # Check if files exist
            old_fg_path = os.path.join(fg_dir, f"{old_stem}_fg.npy")
            old_flow_path = os.path.join(flow_dir, f"{old_stem}_flow.npy")
            
            new_fg_path = os.path.join(fg_dir, f"{new_stem}_fg.npy")
            new_flow_path = os.path.join(flow_dir, f"{new_stem}_flow.npy")
            
            if os.path.exists(old_fg_path) and os.path.exists(old_flow_path):
                # Rename foreground file
                shutil.move(old_fg_path, new_fg_path)
                print(f"  ✓ Renamed FG: {old_stem}_fg.npy → {new_stem}_fg.npy ({label_name})")
                
                # Rename flow file
                shutil.move(old_flow_path, new_flow_path)
                print(f"  ✓ Renamed Flow: {old_stem}_flow.npy → {new_stem}_flow.npy ({label_name})")
                
                # Update CSV row
                updated_rows.append({
                    'filename': new_stem,
                    'label': label
                })
                renamed_files.append((old_stem, new_stem, label_name))
            else:
                print(f"  ⚠️  Files not found for {old_stem} in {label_name} directory")
                # Keep original name in CSV
                updated_rows.append({
                    'filename': old_stem,
                    'label': label
                })
    
    # Update the dataframe
    # Remove old duplicate rows
    df = df[~df['filename'].isin(duplicate_groups.groups.keys())]
    
    # Add updated rows
    updated_df = pd.DataFrame(updated_rows)
    df = pd.concat([df, updated_df], ignore_index=True)
    
    # Save updated CSV
    backup_csv = labels_csv.replace('.csv', '_backup.csv')
    shutil.copy(labels_csv, backup_csv)
    print(f"\n💾 Backed up original CSV to: {backup_csv}")
    
    df.to_csv(labels_csv, index=False)
    print(f"✅ Updated CSV saved with {len(df)} entries")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Total duplicates processed: {len(duplicate_groups)}")
    print(f"  Files renamed: {len(renamed_files)}")
    print(f"  Final unique entries in CSV: {len(df)}")
    
    # Show renamed files
    if renamed_files:
        print(f"\n📝 Renamed files:")
        for old, new, label_type in renamed_files[:20]:  # Show first 20
            print(f"  {old} → {new} ({label_type})")
        if len(renamed_files) > 20:
            print(f"  ... and {len(renamed_files) - 20} more")
    
    print("\n✅ Done! Now re-split the CSV using split_labels_csv.py")

if __name__ == "__main__":
    fix_duplicate_filenames()