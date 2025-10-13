"""
Utility script to update class labels in dataset using a trained classifier.
Useful for creating multitask datasets from keypoint-only datasets.
"""

import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import timm


def classify_image(image, model, device, img_size=224, grayscale=False):
    """Classify a single image.
    
    Args:
        image: Image array (BGR)
        model: PyTorch model
        device: Device to run on
        img_size: Input size for model
        grayscale: Whether to convert to grayscale
        
    Returns:
        Predicted class ID
    """
    # Preprocess
    image_resized = cv2.resize(image, (img_size, img_size))
    
    if grayscale:
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_resized = np.stack([gray] * 3, axis=-1)
    
    # To tensor
    input_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
    input_tensor = input_tensor.float().to(device) / 255.0
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Handle different output formats
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        if outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)
        
        predicted_class = outputs.argmax(dim=1).item()
    
    return predicted_class


def update_dataset_labels(args):
    """Update class labels in dataset using classifier."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load classifier model
    if args.model_type == 'timm':
        model = timm.create_model(
            args.model_name,
            pretrained=False,
            num_classes=args.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Loaded model: {args.model_name}")
    
    # Process dataset
    images_root = Path(args.images_dir)
    labels_root = Path(args.labels_dir)
    
    # Find all subdirectories
    subdirs = []
    for subdir in images_root.iterdir():
        if subdir.is_dir():
            subdirs.append(subdir.name)
    
    if not subdirs:
        subdirs = ['.']
    
    print(f"Found {len(subdirs)} subdirectories")
    
    total_updated = 0
    
    for subdir in tqdm(subdirs, desc="Processing directories"):
        img_dir = images_root / subdir
        lbl_dir = labels_root / subdir
        
        if not lbl_dir.exists():
            print(f"Warning: No labels directory for {subdir}")
            continue
        
        # Process each image
        for img_path in tqdm(list(img_dir.rglob("*.*")), desc=f"  {subdir}", leave=False):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # Find corresponding label
            rel_path = img_path.relative_to(img_dir)
            label_path = lbl_dir / rel_path.with_suffix('.txt')
            
            if not label_path.exists():
                print(f"Warning: No label for {img_path}")
                continue
            
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Cannot read {img_path}")
                continue
            
            # Classify
            try:
                predicted_class = classify_image(
                    image, model, device,
                    img_size=args.img_size,
                    grayscale=args.grayscale
                )
            except Exception as e:
                print(f"Error classifying {img_path}: {e}")
                continue
            
            # Read label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
                continue
            
            if not lines:
                continue
            
            # Update class ID (first token)
            parts = lines[0].strip().split()
            if not parts:
                continue
            
            old_class = parts[0]
            new_line = f"{predicted_class} {' '.join(parts[1:])}\n"
            lines[0] = new_line
            
            # Write back
            try:
                with open(label_path, 'w') as f:
                    f.writelines(lines)
                
                total_updated += 1
                
                if args.verbose:
                    print(f"Updated {label_path}: {old_class} -> {predicted_class}")
                    
            except Exception as e:
                print(f"Error writing {label_path}: {e}")
    
    print(f"\nCompleted! Updated {total_updated} labels")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update dataset labels using classifier')
    
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Images directory')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Labels directory')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to classifier model weights')
    parser.add_argument('--model_type', type=str, default='timm',
                        choices=['timm'])
    parser.add_argument('--model_name', type=str, default='mobilenetv3_small_100',
                        help='Model architecture name')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--grayscale', action='store_true',
                        help='Convert to grayscale')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each update')
    
    args = parser.parse_args()
    update_dataset_labels(args)

