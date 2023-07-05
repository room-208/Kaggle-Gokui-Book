from torch import nn


def create_model():
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
        nn.ReLU(),
        nn.Linear(16, 10),
    )
    return model
