# src/gradcam.py

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from train import FashionCNN, DEVICE  # import the model з train.py
import os

# ----- Loading the model -----
model = FashionCNN().to(DEVICE)
model.load_state_dict(torch.load("models/fashion_cnn.pt", map_location=DEVICE))  # or from train save
model.eval()

# ----- Loading an example -----
transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
classes = test_set.classes

# ----- Grad-CAM for the last Conv layer -----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, target_class]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()

# ----- Visualization -----
def show_gradcam(image_tensor, heatmap, label, prediction, idx):
    image = image_tensor.squeeze().cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # heat map overlay
    plt.title(f"True: {classes[label]}, Pred: {classes[prediction]}")
    plt.axis('off')
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/gradcam_{idx}.png")
    plt.close()

# ----- Home -----
def main():
    gradcam = GradCAM(model, model.conv2)
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        heatmap = gradcam.generate_cam(images, predicted.item())
        show_gradcam(images[0], heatmap, labels.item(), predicted.item(), idx)

        if idx >= 4:  # we will keep only the first 5
            break

if __name__ == "__main__":
    main()
