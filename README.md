# 🧠 Explainable AI: Fashion-MNIST Classifier with Grad-CAM

This project demonstrates a simple but powerful CNN-based image classifier with explainability using Grad-CAM, applied to the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

---

## ✨ Features

- ✅ CNN classifier with ~91% test accuracy
- 🔥 Grad-CAM visualization for model interpretability
- 🎛️ Gradio web interface for easy testing
- 🎲 Random test sample button
- 📦 Clean project structure (src/, app/, models/, outputs/)

---

## 🧪 Quick Start

```bash
git clone https://github.com/JustIsVibe/explainable-ai-fashionmnist.git
cd explainable-ai-fashionmnist
pip install -r requirements.txt

# Train the model
python src/train.py

# Launch web interface
python app/gradio_app.py
