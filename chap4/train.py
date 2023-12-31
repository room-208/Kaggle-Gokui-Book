from pathlib import Path

import torch
import torch.nn as nn
from angular_model import AngularModel
from augmentation import get_augmentations
from gldv2dataset import get_dataloaders
from metrics import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import save_checkpoint


def train(
    path_train_csv: str,  # "gldv2_micro/train.csv"
    path_val_csv: str,  # "gldv2_micro/val.csv"
    outputs_dir: str,  # "outputs"
    gldv2_micro_path: str,
    input_size: int = 128,
    num_epochs: int = 10,
    batch_size: int = 128,
    num_workers: int = 12,
    backbone: str = "resnet34",
    init_lr: float = 0.001,
    device: str = "cuda",
):
    train_transform, val_transform = get_augmentations(input_size)
    dataloaders = get_dataloaders(
        path_train_csv,
        path_val_csv,
        gldv2_micro_path,
        train_transform,
        val_transform,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
    )

    model = AngularModel(
        n_classes=dataloaders["train"].dataset.n_classes,
        model_name=backbone,
        pretrained=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    ttl_iters = num_epochs * len(dataloaders["train"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=0.001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(num_epochs):
        train_meters = {
            "loss": AverageMeter(),
            "acc": AverageMeter(),
        }
        model.train()
        for iter_idx, (X, y) in enumerate(dataloaders["train"]):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            X_out = model(X, y)
            loss = criterion(X_out, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

            trn_loss = loss.item()
            trn_acc = accuracy(X_out, y)

            train_meters["loss"].update(trn_loss, n=X.size(0))
            train_meters["acc"].update(trn_acc, n=X.size(0))

            if iter_idx % 100 == 0:
                print(
                    "Epoch {:.4f} / trn/loss={:.4f}, trn/acc={:.4f}".format(
                        epoch + iter_idx / len(dataloaders["train"]),
                        train_meters["loss"].avg,
                        train_meters["acc"].avg,
                    )
                )

        val_meters = {
            "loss": AverageMeter(),
            "acc": AverageMeter(),
        }
        model.eval()
        for iter_idx, (X, y) in enumerate(dataloaders["val"]):
            X, y = X.to(device), y.to(device)

            with torch.no_grad():
                X_out = model(X, y)
                loss = criterion(X_out, y)

                val_loss = loss.item()
                val_acc = accuracy(X_out, y)

            val_meters["loss"].update(val_loss, n=X.size(0))
            val_meters["acc"].update(val_acc, n=X.size(0))

        print(
            "Ep {:d} / val/loss={:.4f}, val/acc={:.4f}".format(
                epoch + 1, val_meters["loss"].avg, val_meters["acc"].avg
            )
        )

        save_checkpoint(model, epoch + 1, Path(outputs_dir, "arcface_last.pth"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_train_csv", required=True)
    parser.add_argument("--path_val_csv", required=True)
    parser.add_argument("--outputs_dir", required=True)
    parser.add_argument("--gldv2_micro_path", required=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    path_train_csv = args.path_train_csv
    path_val_csv = args.path_val_csv
    outputs_dir = args.outputs_dir
    gldv2_micro_path = args.gldv2_micro_path
    batch_size = args.batch_size
    device = args.device

    train(
        path_train_csv=path_train_csv,
        path_val_csv=path_val_csv,
        outputs_dir=outputs_dir,
        gldv2_micro_path=gldv2_micro_path,
        input_size=256,
        batch_size=batch_size,
        device=device,
    )
