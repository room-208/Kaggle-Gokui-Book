import matplotlib.pyplot as plt

from dataloader import create_dataloder_ver1
from device import create_device
from loss import create_lossfn
from model import create_model
from optimizer import create_optimizer
from seed import seed_torch
from train import train_1epoch

if __name__ == "__main__":
    seed_torch()
    device = create_device()
    lossfn = create_lossfn()
    dataloader = create_dataloder_ver1()
    model = create_model()
    optimizer = create_optimizer(model)

    accs = []
    losses = []
    for e in range(100):
        acc, loss = train_1epoch(model, dataloader, lossfn, optimizer, device)
        accs.append(acc)
        losses.append(loss)
    
    plt.plot(accs)
    plt.show()
    
    plt.plot(losses)
    plt.show()