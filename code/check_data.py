import os
import sys
import torch
import numpy as np
from collections import Counter

# 将当前目录加入路径以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import ISIC2017Dataset, ISIC2017_CLASSES
from data_loader import get_isic2017_loaders

def check_dataset_split(name, dataset):
    print(f"\n--- Checking {name} Split ---")
    if len(dataset) == 0:
        print(f"  [!] {name} dataset is empty!")
        return

    print(f"  Total samples: {len(dataset)}")
    
    all_labels = []
    missing_labels = []
    
    for i in range(len(dataset)):
        img_path = dataset.image_paths[i]
        img_id = dataset._get_image_id(img_path)
        label = dataset.labels.get(img_id, -1)
        all_labels.append(label)
        
        if label == -1:
            missing_labels.append(img_path)
            
    label_counts = Counter(all_labels)
    
    print("  Label distribution:")
    for label, count in sorted(label_counts.items()):
        status = ""
        if label == -1:
            status = "  [!!!] MISSING LABEL"
        elif label >= len(ISIC2017_CLASSES):
            status = f"  [!!!] INVALID LABEL (Max expected: {len(ISIC2017_CLASSES)-1})"
        
        label_name = ISIC2017_CLASSES[label] if 0 <= label < len(ISIC2017_CLASSES) else "Unknown"
        print(f"    Class {label} ({label_name}): {count} samples{status}")

    if missing_labels:
        print(f"  [!] Found {len(missing_labels)} images without ground truth labels.")
        print(f"      Example missing: {os.path.basename(missing_labels[0])}")
    else:
        print("  [✓] All images have corresponding labels.")

def main():
    # 默认数据路径（根据你的项目结构）
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    print(f"Base data directory: {base_data_dir}")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(base_data_dir, split)
        if not os.path.exists(split_dir):
            print(f"\n[!] Directory not found: {split_dir}")
            continue
            
        try:
            # 初始化数据集（不应用数据增强以便快速检查）
            dataset = ISIC2017Dataset(data_dir=split_dir, is_training=False)
            check_dataset_split(split, dataset)
        except Exception as e:
            print(f"\n[!] Error checking {split}: {e}")

if __name__ == "__main__":
    main()