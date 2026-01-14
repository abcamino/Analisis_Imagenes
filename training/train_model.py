"""
Training script for aneurysm detection model.
Requires Python 3.11/3.12 with PyTorch and timm.

Usage:
    python train_model.py --data_dir data/processed --epochs 50
"""

import argparse
import os
from pathlib import Path
import yaml
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import timm
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import cv2
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error: {e}")
    print("\nThis script requires Python 3.11 or 3.12 with PyTorch.")
    print("Please create a separate environment:")
    print("  py -3.11 -m venv training_env")
    print("  training_env\\Scripts\\activate")
    print("  pip install torch torchvision timm opencv-python-headless")
    TORCH_AVAILABLE = False


class AneurysmDataset(Dataset):
    """Dataset for aneurysm classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot load image: {img_path}")

        # Resize to 224x224
        image = cv2.resize(image, (224, 224))

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Convert to 3-channel (for pretrained models)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Apply additional transforms
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def create_model(num_classes=2, pretrained=True):
    """Create MobileNetV3 model for classification."""
    model = timm.create_model(
        'mobilenetv3_large_100',
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total


def load_dataset(data_dir):
    """
    Load dataset from directory structure:
    data_dir/
        normal/
            image1.jpg
            ...
        aneurysm/
            image1.jpg
            ...
    """
    data_dir = Path(data_dir)
    image_paths = []
    labels = []

    # Load normal images (label 0)
    normal_dir = data_dir / 'normal'
    if normal_dir.exists():
        for img_path in normal_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(img_path)
                labels.append(0)

    # Load aneurysm images (label 1)
    aneurysm_dir = data_dir / 'aneurysm'
    if aneurysm_dir.exists():
        for img_path in aneurysm_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(img_path)
                labels.append(1)

    print(f"Loaded {len(image_paths)} images:")
    print(f"  - Normal: {labels.count(0)}")
    print(f"  - Aneurysm: {labels.count(1)}")

    return image_paths, labels


def main():
    if not TORCH_AVAILABLE:
        return

    parser = argparse.ArgumentParser(description='Train aneurysm detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='models/', help='Output directory')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    image_paths, labels = load_dataset(args.data_dir)

    if len(image_paths) < 10:
        print("\nWarning: Very few images for training.")
        print("Consider using data augmentation or obtaining more data.")
        print("Minimum recommended: 100+ images per class")

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(f"\nTraining set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")

    # Create datasets
    train_dataset = AneurysmDataset(train_paths, train_labels)
    val_dataset = AneurysmDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model with val_acc: {val_acc*100:.2f}%")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    print("\nTo export to ONNX, run:")
    print(f"  python export_onnx.py --checkpoint {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()
