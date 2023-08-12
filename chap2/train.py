import torch
from tqdm import tqdm


def train_1epoch(model, train_loader, lossfn, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(train_loader, total=len(train_loader)):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = lossfn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * x.size(0)
        _, pred = torch.max(out, dim=1)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss
