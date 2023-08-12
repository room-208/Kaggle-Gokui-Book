from torch import nn


def setup_lossfn():
    return nn.CrossEntropyLoss()
