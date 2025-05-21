import torch

import torch.nn as nn
from torchvision import models

class XRayClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.densenet121(pretrained=True)
        
        # Freeze base layers
        for param in self.base.parameters():
            param.requires_grad = False
            
        # Replace classifier
        num_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = XRayClassifier().to(device)
