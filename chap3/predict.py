import numpy as np
import torch
from tqdm import tqdm

from dataloader import set_transform
from tta import setup_tta_transforms


def predict(model, loader, device):
    pred_fun = torch.nn.Softmax(dim=1)
    preds = []
    with torch.no_grad():
        for x, _ in tqdm(loader, total=len(loader)):
            x = x.to(device)
            out = model(x)
            y = pred_fun(out)
            y = y.cpu().numpy()
            y = y[:, 1]
            preds.append(y)
    preds = np.concatenate(preds)
    return preds


def predict_tta(model, loader, device):
    tta_transforms = setup_tta_transforms()
    tta_preds = []
    for transform in tta_transforms:
        set_transform(loader.dataset, transform)
        preds = predict(model, loader, device)
        tta_preds.append(preds)
    tta_preds = np.mean(tta_preds, axis=0)
    return tta_preds
