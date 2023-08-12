from torch import optim


def create_optimizer(model):
    return optim.SGD(model.parameters(), lr=1e-2)
