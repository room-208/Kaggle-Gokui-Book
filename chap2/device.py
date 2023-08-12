import torch


def create_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
