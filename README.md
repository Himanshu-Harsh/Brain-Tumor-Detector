<div align="center">
  <img src="https://img.icons8.com/color/150/000000/brain.png" alt="Brain Icon" width="100"/>
  <h1>NeuroScan Classifier</h1>
  <p><b>Advanced Brain Tumor MRI Classification with CBAM Attention and ONNX Runtime</b></p>

  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX">
    <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
    <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5">
  </p>
</div>

<br/>

## 🧠 Overview

NeuroScan is an end-to-end machine learning project designed to classify brain MRI scans into four unique categories: **Glioma, Meningioma, Pituitary Tumor, and Healthy (No Tumor)**. 

The project encompasses a custom deep learning architecture trained in **PyTorch**, exported to **ONNX**, and deployed via an ultra-sleek, interactive, and client-side **Vanilla Web Frontend** using `onnxruntime-web`. Because the inference happens entirely in the browser, your data never leaves your device!

---

## ✨ Key Features

- **Custom State-of-the-art CNN**: Features multi-scale feature fusion across Residual blocks.
- **CBAM Attention Mechanism**: Uses Convolutional Block Attention Module (channel + spatial attention) to accurately locus on tumor regions.
- **Modern Glassmorphism UI**: A beautiful, dark-mode web application featuring animated confidence distributions and drag-and-drop parsing.
- **Privacy-First Inference**: Runs 100% locally in the browser. Zero server uploads.
- **Explainable AI (Grad-CAM)**: Generates heatmaps during evaluation to visualize exactly *where* the model is looking.

---

## 🚀 Quick Start

### 1. Web Application (Inference)

You can run the web application locally without needing Python or external dependencies (other than a modern browser).

```bash
# Start a simple HTTP server in this directory
python -m http.server 8000
```
Then navigate to `http://localhost:8000/` in your browser.

> [!TIP]
> Drag and drop an MRI scan from the `/data/Testing` folder directly into the web interface to see the inference in real-time.

### 2. Model Training (PyTorch)

If you'd like to train the model from scratch, ensure you have PyTorch installed.

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pillow
```

Run the training script. It handles dataset splitting, training, mixup augmentations, evaluation, and generating Grad-CAM plots.

```bash
python train.py
```

<details>
<summary><b>Click to expand: Training Configuration & Hyperparameters</b></summary>
<br/>

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 70 | With Early Stopping (patience=18) |
| **Batch Size** | 24 | - |
| **Learning Rate** | `2e-4` | With Cosine Annealing and Warmup |
| **Weight Decay** | `2e-4` | AdamW optimizer |
| **Loss Function** | Focal Loss | `gamma=2.0` to handle hard samples |
| **Augmentations** | Mixup, Erasing | Random Affine, ColorJitter, Horizontal Flip |

</details>

---

## 🧬 Architecture Details

<details>
<summary><b>Click to expand: BrainTumorNet Architecture</b></summary>
<br/>
The custom `BrainTumorNet` is built to maximize feature extraction while keeping parameters low for web deployment:

1. **Stem**: Initial `ConvBN` block for spatial dimensionality reduction.
2. **Residual Blocks**: 4 stages of `ResBlocks` infused with Convolutional Block Attention Modules (**CBAM**).
3. **Multi-Scale Fusion**: Concatenates identical Adaptive Average and Max pools from both Stage 3 and Stage 4.
4. **Classifier Head**: Contains heavily regularized `Linear` layers with Dropout and `GELU` activations mapping out to the 4 classes.

</details>

---

## 📊 Dataset

The model is trained on the Kaggle [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

It applies a dynamic preprocessing technique that automatically crops out large, irrelevant black borders around the brain scans before normalization.

---

## 🛠️ File Structure

```text
📁 brain_new/
├── 📄 index.html                # The sleek Glassmorphism web application
├── 📄 train.py                  # PyTorch model definition and training loop
├── 📄 convert_to_onnx.py        # Script to export the .pth model to ONNX format
├── 📄 brain_tumor_model.onnx    # The compiled model used by the web frontend
├── 📁 data/                     # Training and testing datasets
│   ├── 📁 Training/
│   └── 📁 Testing/
└── 📁 outputs/                  # Auto-generated plots (Curves, Confusion Matrix, Grad-CAM)
```

<br/>

<div align="center">
  <sub>Built for educational and research purposes. Do not use for clinical diagnostics.</sub>
</div>
