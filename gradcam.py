import torch
import cv2
import numpy as np
from bmodel import XRayClassifier  # Import your model class

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, class_idx=None):
        # Forward pass
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        
        # Generate CAM
        weights = self.gradients.mean(dim=(2,3), keepdims=True)
        cam = (weights * self.activations).sum(dim=1, keepdims=True)
        cam = torch.relu(cam)
        cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()

# Initialize model and load trained weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = XRayClassifier().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()  # Set to evaluation mode

# Initialize Grad-CAM with the last convolutional layer
grad_cam = GradCAM(model, model.base.features[-1])
