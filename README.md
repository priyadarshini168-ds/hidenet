# HideNet: Adversarial Image Steganography

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red)](https://streamlit.io)

> Hiding secret images inside cover images using an Attention-Based U-Net with adversarial steganalysis detection.

---

## Overview

HideNet is a deep learning-based image steganography system that hides a secret image invisibly inside a cover image. The system uses a GAN-style adversarial pipeline with three components:

- **HideNet** — Attention-based U-Net encoder that embeds the secret image into the cover image
- **RevealNet** — CNN decoder that extracts the hidden secret from the stego image
- **SteganalysisDetector** — Binary CNN classifier that detects whether an image contains hidden content

---

## Architecture
Cover Image ──┐
├──► HideNet (U-Net + Attention) ──► Stego Image
Secret Image ─┘                                        │
▼
RevealNet (CNN) ──► Revealed Secret
│
Detector (CNN) ──► Hidden? Yes/No
---

## Features

- Attention gates in U-Net skip connections for better feature focus
- GAN-style adversarial training — HideNet learns to fool the detector
- Combined loss: SSIM + MSE for hiding quality, MSE for reveal accuracy
- Interactive Streamlit demo with Encode, Decode and Detect tabs
- Trained on DIV2K (cover) and COCO (secret) datasets

---

## Results

| Metric | Value |
|--------|-------|
| Training Epochs | 37 |
| Best Total Loss | 0.0451 |
| Hide Loss (SSIM+MSE) | ~0.035 |
| Reveal Loss (MSE) | ~0.018 |
| Dataset | DIV2K + COCO (800 images each) |

---

## Project Structure
HideNet/
├── src/
│   ├── hidenet.py       # Attention U-Net encoder
│   ├── revealnet.py     # CNN decoder
│   ├── detector.py      # Steganalysis classifier
│   ├── losses.py        # Custom loss functions
│   ├── pipeline.py      # Combined training pipeline
│   └── dataset.py       # Data loading and preprocessing
├── app/
│   └── streamlit_app.py # Interactive web demo
├── train.py             # Training script (Colab GPU)
└── requirements.txt     # Dependencies
---

## Installation

```bash
git clone https://github.com/priyadarshini168-ds/hidenet.git
cd hidenet
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

> Note: Place trained model weights (`hidenet.h5`, `revealnet.h5`, `detector.h5`) inside the `models/` folder.

---

## Training

Training was done on Google Colab with Tesla T4 GPU using DIV2K and COCO datasets.

```bash
python train.py
```

---

## Tech Stack

- TensorFlow / Keras
- NumPy, OpenCV, scikit-image
- Streamlit
- Google Colab (GPU training)
- DIV2K + COCO datasets

---

## Author

**Priya Darshini**  
Data Analytics | Deep Learning | Computer Vision  
[GitHub](https://github.com/priyadarshini168-ds)
