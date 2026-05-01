import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from model import load_model
from utils import get_logger

# Mapping from disease label to a short description
DISEASE_INFO = {
    "Potato___healthy": "A healthy potato leaf shows no visible lesions, discoloration, or wilting. This is the normal state for cultivated potato plants.",
    "Potato___Early_blight": "Early blight appears as small brownish spots on the leaf surface that expand and develop concentric rings, often surrounded by a yellow halo. It can reduce photosynthetic area and yield if untreated.",
    "Potato___Late_blight": "Late blight causes water‑soaked lesions that turn dark brown or black, often with a rapid spread across the leaf. It is a serious disease that can lead to crop loss quickly.",
}

# Path to the label map generated during data preparation
LABELS_JSON = Path(__file__).resolve().parent / "export" / "labels.json"

def load_labels(path: Path) -> dict:
    """Load the JSON label map (full class name → integer)."""
    with open(path, "r") as f:
        return json.load(f)

def preprocess_image(image_path: str) -> torch.Tensor:
    """Resize, normalize and convert an image to a tensor suitable for the model."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # shape: (1, C, H, W)

def predict(image_path: str, model_path: str, device: str = "cpu") -> Tuple[str, float, str]:
    """Run inference on a single image.
    Returns:
        label (str): Human‑readable class name.
        confidence (float): Probability between 0 and 1.
        description (str): Short disease description from ``DISEASE_INFO``.
    """
    logger = get_logger()
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, device=device)
    labels = load_labels(LABELS_JSON)
    idx_to_label = {int(v): k for k, v in labels.items()}
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)
    label = idx_to_label[int(pred_idx)]
    confidence = float(prob)
    description = DISEASE_INFO.get(label, "No description available.")
    return label, confidence, description

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single potato leaf image")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="export/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--detail", action="store_true", help="Print disease description")
    args = parser.parse_args()
    label, confidence, description = predict(args.image, args.model, args.device)
    print(f"Predicted: {label} ({confidence*100:.2f}% confidence)")
    if args.detail:
        print("Description:", description)

if __name__ == "__main__":
    main()
