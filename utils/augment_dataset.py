"""
Data augmentation script for hand keypoint datasets.
Supports YOLO-Pose format with keypoints.
"""

import os
import argparse
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
from tqdm import tqdm


def read_yolo_label(label_path, img_w, img_h):
    """Read YOLO-Pose format label with keypoints.
    
    Args:
        label_path: Path to label file
        img_w: Image width
        img_h: Image height
        
    Returns:
        cls_id: Class ID
        keypoints: List of (x, y) tuples in pixel coordinates
    """
    with open(label_path, 'r') as f:
        data = list(map(float, f.readline().strip().split()))
    
    cls_id = int(data[0])
    kps = data[5:]  # Skip class_id and bbox
    
    keypoints = []
    num_kps = len(kps) // 3  # Each keypoint has x, y, visibility
    
    for i in range(num_kps):
        x_norm = kps[i * 3]
        y_norm = kps[i * 3 + 1]
        x_px = int(x_norm * img_w)
        y_px = int(y_norm * img_h)
        keypoints.append((x_px, y_px))
    
    return cls_id, keypoints


def write_yolo_label(save_path, class_id, keypoints, img_w, img_h):
    """Write YOLO-Pose format label.
    
    Args:
        save_path: Path to save label
        class_id: Class ID
        keypoints: List of (x, y) tuples in pixel coordinates
        img_w: Image width
        img_h: Image height
    """
    with open(save_path, 'w') as f:
        # Write class_id and dummy bbox (center format)
        line = f"{class_id} 0.5 0.5 0.5 0.5 "
        
        # Write keypoints
        for x_px, y_px in keypoints:
            x_norm = x_px / img_w
            y_norm = y_px / img_h
            line += f"{x_norm:.6f} {y_norm:.6f} 2 "
        
        f.write(line.strip() + '\n')


def get_augmentation_pipeline(aug_type='standard'):
    """Get augmentation pipeline.
    
    Args:
        aug_type: Type of augmentation ('standard', 'light', 'heavy')
        
    Returns:
        Albumentations compose object
    """
    if aug_type == 'light':
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    elif aug_type == 'heavy':
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=50, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OpticalDistortion(distort_limit=0.4, shift_limit=0.3, p=0.3),
            A.CoarseDropout(max_holes=5, max_height=32, max_width=32, p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    else:  # standard
        transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=30, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    return transform


def augment_dataset(args):
    """Augment dataset with keypoints."""
    
    # Setup augmentation
    transform = get_augmentation_pipeline(args.aug_type)
    
    # Prepare output directories
    if os.path.exists(args.output_dir) and not args.append:
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each source directory
    source_dirs = []
    for root, dirs, files in os.walk(args.images_dir):
        if files:  # Has files
            rel_path = os.path.relpath(root, args.images_dir)
            source_dirs.append(rel_path)
    
    if not source_dirs:
        source_dirs = ['.']
    
    print(f"Found {len(source_dirs)} directories to process")
    
    total_augmented = 0
    
    for src_dir in tqdm(source_dirs, desc="Processing directories"):
        images_path = os.path.join(args.images_dir, src_dir)
        labels_path = os.path.join(args.labels_dir, src_dir)
        
        output_images_path = os.path.join(args.output_dir, 'images', src_dir)
        output_labels_path = os.path.join(args.output_dir, 'labels', src_dir)
        
        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_labels_path, exist_ok=True)
        
        # Get image files
        image_files = [f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in tqdm(image_files, desc=f"  {src_dir}", leave=False):
            img_path = os.path.join(images_path, img_file)
            label_path = os.path.join(labels_path, Path(img_file).stem + '.txt')
            
            if not os.path.exists(label_path):
                print(f"Warning: Label not found for {img_file}")
                continue
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Cannot read image {img_file}")
                continue
            
            h, w = image.shape[:2]
            
            # Read label
            try:
                cls_id, keypoints = read_yolo_label(label_path, w, h)
            except Exception as e:
                print(f"Error reading label {label_path}: {e}")
                continue
            
            # Save original
            if args.save_original:
                original_img_save = os.path.join(output_images_path, img_file)
                original_lbl_save = os.path.join(output_labels_path, Path(img_file).stem + '.txt')
                cv2.imwrite(original_img_save, image)
                write_yolo_label(original_lbl_save, cls_id, keypoints, w, h)
            
            # Generate augmented versions
            for i in range(args.num_augmentations):
                try:
                    transformed = transform(image=image, keypoints=keypoints)
                    transformed_image = transformed['image']
                    transformed_keypoints = transformed['keypoints']
                    
                    # Save augmented image
                    aug_img_name = f"{Path(img_file).stem}_aug{i}{Path(img_file).suffix}"
                    aug_img_path = os.path.join(output_images_path, aug_img_name)
                    cv2.imwrite(aug_img_path, transformed_image)
                    
                    # Save augmented label
                    aug_lbl_name = f"{Path(img_file).stem}_aug{i}.txt"
                    aug_lbl_path = os.path.join(output_labels_path, aug_lbl_name)
                    write_yolo_label(
                        aug_lbl_path, cls_id, transformed_keypoints,
                        transformed_image.shape[1], transformed_image.shape[0]
                    )
                    
                    total_augmented += 1
                    
                except Exception as e:
                    print(f"Error augmenting {img_file} (aug {i}): {e}")
                    continue
    
    print(f"\nAugmentation completed!")
    print(f"Total augmented images: {total_augmented}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment hand keypoint dataset')
    
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Input images directory')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Input labels directory')
    parser.add_argument('--output_dir', type=str, default='augmented_dataset',
                        help='Output directory')
    parser.add_argument('--num_augmentations', type=int, default=10,
                        help='Number of augmented versions per image')
    parser.add_argument('--aug_type', type=str, default='standard',
                        choices=['light', 'standard', 'heavy'],
                        help='Type of augmentation')
    parser.add_argument('--save_original', action='store_true',
                        help='Save original images in output')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing output directory')
    
    args = parser.parse_args()
    augment_dataset(args)

