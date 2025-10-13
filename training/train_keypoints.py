"""
Training script for hand keypoint detection.
Supports hard negative mining and multiple architectures.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm, trange
import numpy as np
import cv2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class HandKeypointsDataset(Dataset):
    """Dataset for hand keypoints detection."""
    
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

        # Read keypoints
        with open(label_path, 'r') as f:
            data = list(map(float, f.readline().strip().split()))
        
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

        return image, keypoints_tensor.view(-1, 2), 1.0


class RepeatedHardDataset(torch.utils.data.Dataset):
    """Dataset that repeats hard samples for hard negative mining."""
    
    def __init__(self, base_dataset, hard_samples, repeat_times=3, loss_weight=2.0):
        self.base_dataset = base_dataset
        self.hard_samples = hard_samples
        self.repeat_times = repeat_times
        self.loss_weight = loss_weight
        self.len_base = len(base_dataset)
        self.len_hard = len(hard_samples) * repeat_times

    def __len__(self):
        return self.len_base + self.len_hard

    def __getitem__(self, idx):
        if idx < self.len_base:
            img, kp, _ = self.base_dataset[idx]
            return img, kp, 1.0
        else:
            hard_idx = (idx - self.len_base) % len(self.hard_samples)
            img, kp = self.hard_samples[hard_idx]
            return img.clone(), kp.clone(), self.loss_weight


class HardSampleManager:
    """Manager for hard negative mining."""
    
    def __init__(self, top_percent=0.2, repeat_times=3, loss_weight=2.0):
        self.top_percent = top_percent
        self.repeat_times = repeat_times
        self.loss_weight = loss_weight
        self.hard_samples = []

    def update(self, all_losses, all_images, all_keypoints):
        """Update list of hardest samples."""
        indexed_losses = list(enumerate(all_losses))
        indexed_losses.sort(key=lambda x: x[1], reverse=True)

        num_hard = int(len(all_losses) * self.top_percent)
        top_indices = [i for i, _ in indexed_losses[:num_hard]]

        self.hard_samples = [(all_images[i], all_keypoints[i]) for i in top_indices]

    def get_dataset(self, base_dataset):
        """Create dataset with repeated hard samples."""
        return RepeatedHardDataset(
            base_dataset, self.hard_samples, 
            self.repeat_times, self.loss_weight
        )


def visualize_predictions(images, pred_kps, true_kps, output_dir, epoch, batch_idx, num_samples=2):
    """Visualize keypoint predictions."""
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    indices = random.sample(range(len(images)), k=min(num_samples, len(images)))
    selected_images = images[indices].cpu().numpy()
    selected_preds = pred_kps[indices].cpu().numpy()
    selected_gt = true_kps[indices].cpu().numpy()

    for i, idx in enumerate(indices):
        img = selected_images[i].transpose(1, 2, 0)
        pred_pts = selected_preds[i].reshape(21, 2)
        true_pts = selected_gt[i].reshape(21, 2)

        h, w = img.shape[0], img.shape[1]
        pred_pixels = [(int(x), int(y)) for x, y in pred_pts]
        true_pixels = [(int(x), int(y)) for x, y in true_pts]

        img_to_draw = (img * 255).astype(np.uint8).copy()
        img_gt = img_to_draw.copy()
        img_pred = img_to_draw.copy()

        # Draw keypoints
        for (x, y) in true_pixels:
            cv2.circle(img_gt, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        for (x, y) in pred_pixels:
            cv2.circle(img_pred, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        combined = cv2.addWeighted(img_gt, 0.5, img_pred, 0.5, 0)
        
        save_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_sample_{idx}.jpg")
        cv2.imwrite(save_path, combined[:, :, ::-1])


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                visualize_dir=None, use_hard_mining=False):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    all_losses = []
    all_images = []
    all_keypoints = []

    for batch_idx, (images, keypoints, weights) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        images = images.to(device)
        keypoints = keypoints.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        
        # Forward pass - handle different model outputs
        outputs = model(images)
        if isinstance(outputs, tuple):
            keypoints_preds = outputs[0][:, :, :2] * 256
        else:
            keypoints_preds = outputs[:, :, :2] * 256

        # Calculate weighted loss
        per_sample_loss = ((keypoints_preds - keypoints) ** 2).mean(dim=(1, 2))
        weighted_loss = (per_sample_loss * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()

        train_loss += weighted_loss.item()

        # Store for hard negative mining
        if use_hard_mining:
            with torch.no_grad():
                all_losses.extend(per_sample_loss.cpu().tolist())
                all_images.extend(images.cpu())
                all_keypoints.extend(keypoints.cpu())

        # Visualize
        if visualize_dir and batch_idx % 100 == 0:
            with torch.no_grad():
                visualize_predictions(
                    images, keypoints_preds, keypoints,
                    visualize_dir, epoch, batch_idx
                )

    avg_loss = train_loss / len(train_loader)
    return avg_loss, all_losses, all_images, all_keypoints


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images, keypoints, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                keypoints_pred = outputs[0][:, :, :2] * 256
            else:
                keypoints_pred = outputs[:, :, :2] * 256
            
            loss = criterion(keypoints_pred, keypoints)
            val_loss += loss.item()
    
    avg_loss = val_loss / len(val_loader)
    return avg_loss


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.visualize_dir, exist_ok=True)

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
    train_dataset = HandKeypointsDataset(
        image_dir=args.train_images,
        label_dir=args.train_labels,
        transform=transform
    )
    
    val_dataset = HandKeypointsDataset(
        image_dir=args.val_images,
        label_dir=args.val_labels,
        transform=transform
    )

    # Initialize hard sample manager
    hard_sample_manager = None
    if args.use_hard_mining:
        hard_sample_manager = HardSampleManager(
            top_percent=0.2, 
            repeat_times=3, 
            loss_weight=2.0
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_loss = float('inf')

    # Training loop
    for epoch in trange(args.epochs):
        # Prepare dataloader
        if epoch == 0 or not args.use_hard_mining:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=args.num_workers
            )
        else:
            train_dataset_with_hard = hard_sample_manager.get_dataset(train_dataset)
            train_loader = DataLoader(
                train_dataset_with_hard,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Train
        train_loss, all_losses, all_images, all_keypoints = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            visualize_dir=args.visualize_dir if epoch % 5 == 0 else None,
            use_hard_mining=args.use_hard_mining
        )

        # Update hard samples
        if args.use_hard_mining and len(all_losses) > 0:
            hard_sample_manager.update(all_losses, all_images, all_keypoints)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Log
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Save checkpoint
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"keypoints_epoch{epoch}_trainloss{train_loss:.4f}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best_keypoints.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hand keypoint detector')
    
    # Data
    parser.add_argument('--train_images', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--val_images', type=str, required=True)
    parser.add_argument('--val_labels', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    
    # Model
    parser.add_argument('--model_type', type=str, default='blazehand',
                        choices=['blazehand'])
    parser.add_argument('--pretrained_weights', type=str, default=None)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_hard_mining', action='store_true',
                        help='Enable hard negative mining')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--visualize_dir', type=str, default='results')
    parser.add_argument('--log_dir', type=str, default='runs/keypoint_training')
    
    args = parser.parse_args()
    main(args)

