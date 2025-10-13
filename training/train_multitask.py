"""
Training script for multitask learning: keypoints + gesture classification.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm, trange
import numpy as np
import cv2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class HandMultitaskDataset(Dataset):
    """Dataset for multitask learning: keypoints + gesture classification."""
    
    def __init__(self, image_dir, label_dir, transform=None, num_keypoints=21):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.num_keypoints = num_keypoints

        # Collect all image files
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = os.path.join(
            self.label_dir,
            Path(img_path).relative_to(self.image_dir).with_suffix(".txt")
        )

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.stack([image] * 3, axis=-1)
        h, w = image.shape[:2]

        # Read label
        with open(label_path, 'r') as f:
            data = list(map(float, f.readline().strip().split()))
        
        cls_id = int(data[0])
        kps = data[5:]  # Skip bbox
        
        keypoints = []
        for i in range(self.num_keypoints):
            x = kps[i * 3]
            y = kps[i * 3 + 1]
            keypoints.append((x, y))

        # Convert to pixel coordinates
        kp_pixel = [(int(x * w), int(y * h)) for x, y in keypoints]

        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, keypoints=kp_pixel)
            image = transformed['image'] / 255
            kp_transformed = transformed['keypoints']
            keypoints_tensor = torch.tensor(kp_transformed, dtype=torch.float32)
        else:
            image = transforms.ToTensor()(image)
            keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)

        return image, cls_id, keypoints_tensor.view(-1, 2)


def accuracy(outputs, labels):
    """Calculate accuracy."""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_epoch(model, train_loader, kps_criterion, cls_criterion, 
                optimizer, device, kps_weight=1.0, cls_weight=2.0):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    train_acc = 0

    for images, cls_id, keypoints in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        keypoints = keypoints.to(device)
        cls_id = cls_id.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        cls_pred, keypoints_preds = model(images)
        keypoints_preds = keypoints_preds[:, :, :2] * 256
        
        # Calculate losses
        kps_loss = kps_criterion(keypoints_preds, keypoints)
        cls_loss = cls_criterion(cls_pred, cls_id)
        loss = kps_loss * kps_weight + cls_loss * cls_weight
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(cls_pred, cls_id)

    avg_loss = train_loss / len(train_loader)
    avg_acc = train_acc / len(train_loader)
    
    return avg_loss, avg_acc


def validate(model, val_loader, kps_criterion, cls_criterion, 
             device, kps_weight=1.0, cls_weight=2.0):
    """Validate model."""
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for images, cls_id, keypoints in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            keypoints = keypoints.to(device)
            cls_id = cls_id.to(device)
            
            cls_pred, keypoints_pred = model(images)
            keypoints_pred = keypoints_pred[:, :, :2] * 256
            
            kps_loss = kps_criterion(keypoints_pred, keypoints)
            cls_loss = cls_criterion(cls_pred, cls_id)
            loss = kps_loss * kps_weight + cls_loss * cls_weight
            
            val_loss += loss.item()
            val_acc += accuracy(cls_pred, cls_id)
    
    avg_loss = val_loss / len(val_loader)
    avg_acc = val_acc / len(val_loader)
    
    return avg_loss, avg_acc


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # Augmentations
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    # Load datasets
    train_dataset = HandMultitaskDataset(
        image_dir=args.train_images,
        label_dir=args.train_labels,
        transform=transform
    )
    
    val_dataset = HandMultitaskDataset(
        image_dir=args.val_images,
        label_dir=args.val_labels,
        transform=transform
    )

    # Setup weighted sampler for class imbalance
    if args.use_weighted_sampler:
        train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
        cls_counts = np.bincount(train_labels, minlength=args.num_classes)
        class_weights = len(train_labels) / (len(cls_counts) * cls_counts)
        sample_weights = [class_weights[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load model
    if args.model_type == 'blazehand':
        from MediaPipePyTorch.blazehand_landmark import BlazeHandLandmark
        model = BlazeHandLandmark()
        if args.pretrained_weights:
            state_dict = torch.load(args.pretrained_weights)
            model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model.to(device)

    # Setup training
    kps_criterion = nn.MSELoss()
    
    if args.class_weights:
        weights = torch.tensor(args.class_weights).to(device)
        cls_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    else:
        cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_acc = 0.0
    train_accuracies = []
    epochs_list = []

    # Training loop
    for epoch in trange(args.epochs):
        epochs_list.append(epoch + 1)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, kps_criterion, cls_criterion,
            optimizer, device, args.kps_weight, args.cls_weight
        )
        
        train_accuracies.append(train_acc)

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, kps_criterion, cls_criterion,
            device, args.kps_weight, args.cls_weight
        )

        # Log
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Plot accuracy
        if epoch % 10 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_list, train_accuracies, label='Train Accuracy', marker='o', linewidth=2)
            plt.title('Training Accuracy per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.plot_dir, f"accuracy_epoch_{epoch+1}.png"))
            plt.close()

        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"multitask_epoch{epoch}_valacc{val_acc:.4f}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.checkpoint_dir, "best_multitask.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with val_acc: {val_acc:.4f}")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multitask hand model')
    
    # Data
    parser.add_argument('--train_images', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--val_images', type=str, required=True)
    parser.add_argument('--val_labels', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=5)
    
    # Model
    parser.add_argument('--model_type', type=str, default='blazehand')
    parser.add_argument('--pretrained_weights', type=str, default=None)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--kps_weight', type=float, default=1.0,
                        help='Weight for keypoint loss')
    parser.add_argument('--cls_weight', type=float, default=2.0,
                        help='Weight for classification loss')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                        help='Use weighted sampler for class imbalance')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='Class weights for loss (e.g., 2.0 2.0 1.0 2.0 8.0)')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--plot_dir', type=str, default='plots')
    parser.add_argument('--log_dir', type=str, default='runs/multitask_training')
    
    args = parser.parse_args()
    main(args)

