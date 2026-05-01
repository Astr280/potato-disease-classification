import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import json

from data import prepare_dataset, PotatoDataset
from model import load_model

def compute_val_accuracy():
    device = "cpu"
    model = load_model("export/best_model.pth", device=device)
    model.eval()
    val_imgs, val_labels = prepare_dataset(split="val")
    val_dataset = PotatoDataset(val_imgs, val_labels, train=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    preds = []
    trues = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            _, pred = torch.max(outs, 1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.numpy())
    return accuracy_score(trues, preds)

def test_validation_accuracy():
    acc = compute_val_accuracy()
    assert acc >= 0.90, f"Validation accuracy {acc:.2f} is below 0.90"
