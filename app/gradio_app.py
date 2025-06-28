# app/gradio_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from train import FashionCNN, DEVICE
from gradcam import GradCAM

# ----- Loading the model -----
model = FashionCNN().to(DEVICE)
model.load_state_dict(torch.load("models/fashion_cnn.pt", map_location=DEVICE))
model.eval()

# ----- Grad-CAM -----
gradcam = GradCAM(model, model.conv2)
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ----- Transformation -----
transform = transforms.Compose([
    transforms.Grayscale(),          # important because model expects 1 channel
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# ----- Downloading the test suite -----
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

test_set = FashionMNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

def random_test_sample():
    for image_tensor, _ in test_loader:
        image_pil = transforms.ToPILImage()(image_tensor.squeeze(0))
        return image_pil

# ----- Main function -----
def predict_and_explain(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    label = classes[predicted.item()]

    heatmap = gradcam.generate_cam(image_tensor, predicted.item())

    # Rotate the image with the overlay Grad-CAM
    image_np = np.array(image.resize((28, 28)).convert('L')) / 255.0
    plt.figure(figsize=(3, 3))
    plt.imshow(image_np, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/live_gradcam.png")
    plt.close()

    return label, "outputs/live_gradcam.png"

# ----- Gradio інтерфейс -----
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Fashion-MNIST з Grad-CAM")
    gr.Markdown("Upload your image or click the button for a random one.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Image(28x28)")
        btn_random = gr.Button("🎲 Random image")

    label_output = gr.Label(label="Forecast")
    gradcam_output = gr.Image(type="filepath", label="Grad-CAM")

    btn_random.click(fn=random_test_sample, outputs=image_input)
    image_input.change(fn=predict_and_explain, inputs=image_input, outputs=[label_output, gradcam_output])

demo.launch()
