import numpy as np
import torch
from tqdm import tqdm


def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)
            out = model(x)
            y = pred_fun(out)
            y = y.cpu().numpy()
            preds.append(y)

    preds = np.concatenate(preds)
    return preds
