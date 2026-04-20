#!/usr/bin/env python3
"""
Create 5-fold stratified splits for Task_7 pancreas dataset.
Ensures balanced class distribution across folds.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import nibabel as nib
from collections import Counter

def get_label_properties(label_path):
    """
    Get properties of label file to determine stratification key.
    Uses pancreas+tumor presence as stratification criterion.
    """
    try:
        label = nib.load(label_path).get_fdata()
        has_pancreas = np.any(label == 1)
        has_tumor = np.any(label == 2)
        # Stratify by: 0=neither, 1=pancreas only, 2=both
        if has_pancreas and has_tumor:
            return 2
        elif has_pancreas:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Warning: Could not read {label_path}: {e}")
        return 0


def create_5fold_splits(
    data_dir="/home/minhchau/anaconda3/envs/datn/DATN/dataset/Task_7/imagesTr",
    labels_dir="/home/minhchau/anaconda3/envs/datn/DATN/dataset/Task_7/labelsTr",
    output_dir="/home/minhchau/anaconda3/envs/datn/DATN/monai_benchmark/splits",
    n_splits=5,
    random_seed=42
):
    """
    Create stratified 5-fold CV splits.
    
    Args:
        data_dir: Path to images directory
        labels_dir: Path to labels directory
        output_dir: Path to save fold JSON files
        n_splits: Number of folds (default 5)
        random_seed: For reproducibility
    """
    
    data_dir = Path(data_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(random_seed)
    
    # Get all image files
    image_files = sorted([f for f in data_dir.glob("*.nii.gz")])
    print(f"\n📁 Found {len(image_files)} images in {data_dir}")
    
    if len(image_files) == 0:
        print(f"❌ Error: No images found in {data_dir}")
        return
    
    # Create stratification labels
    print("\n🔍 Analyzing label properties for stratification...")
    stratify_labels = []
    for img_file in image_files:
        label_name = img_file.name.replace("_0000.nii.gz", ".nii.gz")
        label_path = labels_dir / label_name
        
        if label_path.exists():
            label_prop = get_label_properties(label_path)
            stratify_labels.append(label_prop)
        else:
            print(f"⚠️  Warning: Label not found for {img_file.name}, assuming class 0")
            stratify_labels.append(0)
    
    stratify_labels = np.array(stratify_labels)
    
    # Print class distribution
    class_counts = Counter(stratify_labels)
    print(f"\n📊 Label distribution:")
    for cls, count in sorted(class_counts.items()):
        cls_name = ["Neither", "Pancreas Only", "Pancreas+Tumor"][cls]
        print(f"   Class {cls} ({cls_name}): {count} samples")
    
    # Create stratified K-fold splits
    print(f"\n✂️  Creating {n_splits}-fold stratified splits...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    folds_data = {f"fold_{i}": {"train": [], "val": []} for i in range(n_splits)}
    
    fold_idx = 0
    for train_indices, val_indices in skf.split(image_files, stratify_labels):
        train_names = [image_files[i].stem.replace("_0000", "") for i in train_indices]
        val_names = [image_files[i].stem.replace("_0000", "") for i in val_indices]
        
        folds_data[f"fold_{fold_idx}"]["train"] = train_names
        folds_data[f"fold_{fold_idx}"]["val"] = val_names
        
        # Analyze fold distribution
        fold_train_labels = stratify_labels[train_indices]
        fold_val_labels = stratify_labels[val_indices]
        
        print(f"\n  Fold {fold_idx}:")
        print(f"    Train: {len(train_names)} samples")
        for cls in range(3):
            count = np.sum(fold_train_labels == cls)
            if count > 0:
                cls_name = ["Neither", "Pancreas Only", "Pancreas+Tumor"][cls]
                print(f"      Class {cls} ({cls_name}): {count}")
        
        print(f"    Val:   {len(val_names)} samples")
        for cls in range(3):
            count = np.sum(fold_val_labels == cls)
            if count > 0:
                cls_name = ["Neither", "Pancreas Only", "Pancreas+Tumor"][cls]
                print(f"      Class {cls} ({cls_name}): {count}")
        
        fold_idx += 1
    
    # Save fold indices to JSON
    print(f"\n💾 Saving fold indices to {output_dir}...")
    for fold_name, fold_data in folds_data.items():
        fold_file = output_dir / f"{fold_name}.json"
        with open(fold_file, 'w') as f:
            json.dump(fold_data, f, indent=2)
        print(f"   ✓ {fold_file}")
    
    # Verification: Check all images are used exactly once
    all_train = []
    all_val = []
    for fold_data in folds_data.values():
        all_train.extend(fold_data['train'])
        all_val.extend(fold_data['val'])
    
    print(f"\n✅ Verification:")
    print(f"   Total train images: {len(all_train)}")
    print(f"   Total val images: {len(all_val)}")
    print(f"   Total: {len(all_train) + len(all_val)} / {len(image_files)}")
    print(f"   No overlap: {len(set(all_train) & set(all_val)) == 0}")
    
    return folds_data


if __name__ == "__main__":
    folds = create_5fold_splits()
    print("\n" + "="*60)
    print("Phase 1 ✓ Complete: 5-fold CV splits created successfully!")
    print("="*60)
