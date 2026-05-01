import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Create a ResNet18 based model for potato disease classification.
    Args:
        num_classes: number of disease categories.
        freeze_backbone: if True, freeze all layers except the final fully‑connected layer.
    """
    model = models.resnet18(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the final layer parameters
        for param in model.fc.parameters():
            param.requires_grad = True
    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load a checkpoint and return the model in eval mode.
    The checkpoint is expected to contain the state_dict and num_classes.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes")
    if num_classes is None:
        raise ValueError("Checkpoint missing 'num_classes' entry")
    model = build_model(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
