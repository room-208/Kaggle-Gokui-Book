import torchvision
from torch import nn


def create_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model