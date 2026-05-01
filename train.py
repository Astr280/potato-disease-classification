import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from data import download_and_extract, prepare_dataset, PotatoDataset
from model import build_model
from utils import set_seed, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train potato disease classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ResNet backbone during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="export", help="Directory to save checkpoints and label map")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = get_logger()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Ensure data is available
    download_and_extract()

    # Prepare train/val splits
    train_images, train_labels = prepare_dataset(split="train")
    val_images, val_labels = prepare_dataset(split="val")

    train_dataset = PotatoDataset(train_images, train_labels, train=True)
    val_dataset = PotatoDataset(val_images, val_labels, train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(set(train_labels))
    model = build_model(num_classes, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "labels.json"
    # labels.json already saved by data.prepare_dataset; copy if needed
    if not label_path.exists():
        logger.warning("Label map not found in output dir; ensure data preparation saved it.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        logger.info(f"Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        val_acc = val_correct / val_total
        logger.info(f"Val acc: {val_acc:.4f}")

        # Save checkpoint if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "val_acc": val_acc,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        scheduler.step()

if __name__ == "__main__":
    main()
