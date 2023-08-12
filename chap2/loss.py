from torch import nn


def create_lossfn():
    return nn.CrossEntropyLoss()
