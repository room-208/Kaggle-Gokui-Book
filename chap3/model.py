from torchvision.models import resnet50, ResNet50_Weights
from torch import nn


def setup_model(device):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    return model
