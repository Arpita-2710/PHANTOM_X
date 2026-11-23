import torch.nn as nn
from torchvision import models

def build_model(num_classes=2, pretrained=True):
    """ResNet18 transfer-learning classifier for REAL vs AI-GENERATED."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
