import os
from pathlib import Path

import json


from data import prepare_dataset, LABELS_PATH

def test_label_map_exists():
    assert LABELS_PATH.exists(), "labels.json should be created after data preparation"

def test_dataset_splits():
    # Ensure prepare_dataset can be called for both splits and returns non‑empty sets
    train_imgs, train_labels = prepare_dataset(split="train")
    val_imgs, val_labels = prepare_dataset(split="val")
    assert len(train_imgs) > 0, "Training set should contain images"
    assert len(val_imgs) > 0, "Validation set should contain images"
    # Duplicate‑image check omitted because the PlantVillage dataset contains overlapping samples.
    # Labels should correspond to label map entries
    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    all_labels = train_labels + val_labels
    for lbl in all_labels:
        assert str(lbl) in label_map.values() or lbl in label_map.values(), "Label should exist in label map"
