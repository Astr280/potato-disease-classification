import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import json
from pathlib import Path

from model import load_model

# Load label map
LABELS_PATH = Path(__file__).resolve().parent / "export" / "labels.json"
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
idx_to_label = {int(v): k for k, v in label_map.items()}

@st.cache_resource
def get_model():
    model_path = Path(__file__).resolve().parent / "export" / "best_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(model_path), device=device)
    return model, device

model, device = get_model()

st.title("Potato Disease Classification")
st.write("Upload a potato leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob, pred_idx = torch.max(probs, dim=1)
    pred_label = idx_to_label[int(pred_idx)]
    confidence = float(prob)
    st.success(f"**Prediction:** {pred_label} ({confidence*100:.1f}% confidence)")
