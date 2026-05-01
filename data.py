import os
import json
import random
import urllib.request
import tarfile
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Constants
DATA_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.tar.gz"
DATA_ROOT = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
LABELS_PATH = Path(__file__).resolve().parent / "export" / "labels.json"

def download_and_extract(url: str = DATA_URL, extract_to: Path = RAW_DIR) -> None:
    """Download the PlantVillage tarball and extract only potato disease folders.
    The archive contains many crops; we filter for the potato sub‑directories.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    archive_path = extract_to / "plantvillage.tar.gz"
    if not archive_path.exists():
        print(f"Downloading dataset from {url} ...")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print("Archive already downloaded.")
    # Extract
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "Potato" in m.name]
        tar.extractall(path=extract_to, members=members)
    print("Extraction complete.")

def prepare_dataset(split: str = "train", val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Path], List[int]]:
    """Create train/val splits and return list of image paths and integer labels.
    The function expects the raw data to be in RAW_DIR/PlantVillage‑Dataset‑master/Potato/*
    """
    random.seed(seed)
    # Locate potato disease folders
    base_path = RAW_DIR / "PlantVillage-Dataset-master" / "raw" / "color"
    # Filter only potato disease folders (names start with "Potato___")
    potato_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("Potato___")]
    classes = sorted([d.name.split('___')[1] for d in potato_dirs])
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    # Save label map (use original class names with full prefix for consistency)
    full_label_map = {d.name: idx for idx, d in enumerate(potato_dirs)}
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(full_label_map, f, indent=2)
    # Gather images
    all_images = []
    all_labels = []
    for d in potato_dirs:
        imgs = list(d.rglob("*.JPG")) + list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
        all_images.extend(imgs)
        all_labels.extend([full_label_map[d.name]] * len(imgs))
    # Stratified split
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - val_ratio))
    if split == "train":
        return [p for p, _ in combined[:split_idx]], [l for _, l in combined[:split_idx]]
    else:
        return [p for p, _ in combined[split_idx:]], [l for _, l in combined[split_idx:]]

class PotatoDataset(Dataset):
    """PyTorch Dataset for potato leaf images.
    Args:
        images: list of image file paths
        labels: list of integer labels
        train: whether to apply augmentation (default True)
    """
    def __init__(self, images: List[Path], labels: List[int], train: bool = True):
        self.images = images
        self.labels = labels
        self.train = train
        self.transform = self._build_transform()

    def _build_transform(self):
        base = [transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]
        if self.train:
            aug = [transforms.RandomHorizontalFlip()]
            return transforms.Compose(aug + base)
        return transforms.Compose(base)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label
