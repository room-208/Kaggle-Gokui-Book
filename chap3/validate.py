import torch
from tqdm import tqdm


def validate_1epoch(model, val_loader, lossfn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader, total=len(val_loader)):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = lossfn(out.detach(), y)
            _, pred = torch.max(out, 1)
            total_loss += float(loss) * x.size(0)
            total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss
