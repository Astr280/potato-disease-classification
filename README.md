# Potato Disease Classification

This repository provides an end‑to‑end pipeline for classifying potato leaf diseases using PyTorch and a Streamlit web UI.

## Project Structure
```
potato-disease-classification/
│   README.md
│   requirements.txt
│   train.py          # training loop, validation, checkpointing
│   predict.py        # inference helper
│   model.py          # model definition / wrapper
│   data.py           # dataset class, transforms
│   utils.py          # common utilities (seed, logger)
│   streamlit_app.py  # Streamlit UI for inference
│   export/           # saved model weights, label map JSON
└── data/
    └── raw/          # downloaded raw images (git‑ignore)
    └── processed/    # train/val splits after preprocessing
```

## Setup
```bash
# Clone the repo
git clone <repo-url>
cd potato-disease-classification

# Install dependencies
pip install -r requirements.txt
```

## Training
```bash
python train.py --epochs 20 --batch-size 32 --lr 0.001
```

## Inference UI
```bash
streamlit run streamlit_app.py
```

## Notes
- The script will download the PlantVillage dataset automatically if not present.
- GPU is used if available; otherwise training runs on CPU.
- Model checkpoints are saved in `export/`.

## Quick start
```bash
# Install dependencies
pip install -r requirements.txt

# Train (optional – a pretrained checkpoint is already provided)
python train.py --epochs 10

# Run inference on a single image
python predict.py --image path/to/leaf.jpg

# Launch the web UI
streamlit run streamlit_app.py
```

## Docker
```bash
# Build the image
docker build -t potato-disease .
# Run the container (exposes Streamlit on port 8501)
 docker run -p 8501:8501 potato-disease
```

## CI
This repository uses GitHub Actions to automatically run the test suite on every push.
[![CI](https://github.com/your_user/potato-disease-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/your_user/potato-disease-classification/actions/workflows/ci.yml)
