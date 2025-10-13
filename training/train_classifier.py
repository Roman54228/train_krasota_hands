"""
Training script for hand gesture classifier.
Supports multiple architectures and ClearML logging.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clearml import Task


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = torch.nn.functional.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)

        logpt = torch.gather(logpt, dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        pt = torch.gather(pt, dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def accuracy(outputs, labels):
    """Calculate accuracy."""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def log_confusion_matrix(cm, class_names, epoch, logger=None):
    """Plot and log confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (Epoch {epoch + 1})')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    if logger:
        logger.report_figure(
            title="Confusion Matrix",
            series="epoch",
            iteration=epoch + 1,
            figure=plt
        )
    plt.close()


def get_transforms(img_size=256):
    """Get data augmentation transforms."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            shear=5,
            fill=0
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),
    ])


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle different output formats
        if outputs.dim() > 2:
            outputs = outputs.view(outputs.size(0), -1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    return train_loss, train_acc, all_preds, all_labels


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Handle different output formats
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += accuracy(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    return val_loss, val_acc, val_preds, val_labels


def main(args):
    # Initialize ClearML task
    task = Task.init(
        project_name=args.project_name,
        task_name=args.task_name,
        output_uri=True
    )
    logger = task.get_logger()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    transform = get_transforms(args.img_size)
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Load model
    import timm
    model = timm.create_model(
        args.model_name, 
        pretrained=args.pretrained, 
        num_classes=args.num_classes
    )
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    model.to(device)

    # Setup loss and optimizer
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Log hyperparameters
    task.connect({
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "model": args.model_name,
        "loss": "FocalLoss",
        "gamma": args.focal_gamma,
        "alpha": args.focal_alpha,
        "optimizer": "Adam",
        "data_dir": args.data_dir,
        "num_classes": args.num_classes,
    })

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in trange(args.epochs):
        # Train
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Log metrics
        logger.report_scalar("Loss", "Train", iteration=epoch + 1, value=train_loss)
        logger.report_scalar("Loss", "Validation", iteration=epoch + 1, value=val_loss)
        logger.report_scalar("Accuracy", "Train", iteration=epoch + 1, value=train_acc)
        logger.report_scalar("Accuracy", "Validation", iteration=epoch + 1, value=val_acc)

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        log_confusion_matrix(cm, dataset.classes, epoch, logger)

        # Print stats
        print(f'Epoch [{epoch+1}/{args.epochs}] | '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save checkpoint
        model_path = os.path.join(
            args.output_dir, 
            f'classifier_epoch{epoch}_valacc{val_acc:.4f}.pth'
        )
        torch.save(model.state_dict(), model_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.output_dir, 'best_classifier.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with val_acc: {val_acc:.4f}")

    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hand gesture classifier')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of classes')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/validation split ratio')
    
    # Model
    parser.add_argument('--model_name', type=str, default='mobilenetv3_small_100',
                        help='Model architecture from timm')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='Focal loss alpha')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    
    # ClearML
    parser.add_argument('--project_name', type=str, default='hand_gesture_classification',
                        help='ClearML project name')
    parser.add_argument('--task_name', type=str, default='training',
                        help='ClearML task name')
    
    args = parser.parse_args()
    main(args)

