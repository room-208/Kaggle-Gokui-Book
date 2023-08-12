import numpy as np
import torch
from tqdm import tqdm

from validate import validate_1epoch


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
        _, pred = torch.max(out.detach(), dim=1)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def train(model, lossfn, optimizer, train_loader, val_loader, n_epochs, device):
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfn, optimizer, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfn, device)
        print(f"epoch = {epoch}")
        print(f"train_acc = {train_acc}, train_loss = {train_loss}")
        print(f"val_acc = {val_acc}, val_loss = {val_loss}")


def train_1epoch2(model, train_loader, lossfn, optimizer, lr_scheduler, device):
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
        lr_scheduler.step()
        total_loss += float(loss) * x.size(0)
        _, pred = torch.max(out.detach(), dim=1)
        total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def train2(
    model, lossfn, optimizer, lr_scheduler, train_loader, val_loader, n_epochs, device
):
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        train_acc, train_loss = train_1epoch2(
            model, train_loader, lossfn, optimizer, lr_scheduler, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfn, device)
        print(f"epoch = {epoch}")
        print(f"train_acc = {train_acc}, train_loss = {train_loss}")
        print(f"val_acc = {val_acc}, val_loss = {val_loss}")


def train_1epoch_mixup(
    model, train_loader, lossfn, optimizer, lr_scheduler, mixup_alpha, device
):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in tqdm(train_loader, total=len(train_loader)):
        x = x.to(device)
        y = y.to(device)
        lmd = np.random.beta(mixup_alpha, mixup_alpha)
        perm = torch.randperm(x.size(0)).to(device)
        x2 = x[perm, :]
        y2 = y[perm]
        optimizer.zero_grad()
        out = model(lmd * x + (1.0 - lmd) * x2)
        loss = lmd * lossfn(out, y) + (1.0 - lmd) * lossfn(out, y2)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += float(loss) * x.size(0)
        _, pred = torch.max(out.detach(), dim=1)
        total_acc += lmd * torch.sum(pred == y) + (1.0 - lmd) * torch.sum(pred == y2)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss


def train3_mixup(
    model,
    lossfn,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    n_epochs,
    n_mixup_epochs,
    mixup_alpha,
    device,
):
    for epoch in tqdm(range(n_epochs), total=n_epochs):
        if epoch < n_mixup_epochs:
            train_acc, train_loss = train_1epoch_mixup(
                model,
                train_loader,
                lossfn,
                optimizer,
                lr_scheduler,
                mixup_alpha,
                device,
            )
        else:
            train_acc, train_loss = train_1epoch2(
                model, train_loader, lossfn, optimizer, lr_scheduler, device
            )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfn, device)
        print(f"epoch = {epoch}")
        print(f"train_acc = {train_acc}, train_loss = {train_loss}")
        print(f"val_acc = {val_acc}, val_loss = {val_loss}")
