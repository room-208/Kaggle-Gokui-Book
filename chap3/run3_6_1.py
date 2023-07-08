import argparse
import copy
import os

from torch import optim
from torch.utils.data import DataLoader

from augmentation import setup_crop_flip_transform
from dataloader import (set_transform, setup_test_loader,
                        setup_train_val_datasets)
from device import setup_device
from loss import setup_lossfn
from model import setup_model
from predict import predict
from train import train2
from writer import write_prediction_with_clip


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="./")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    dryrun = args.dryrun
    batch_size = args.batch_size
    n_epochs = 2 if dryrun else 10

    device = setup_device()
    model = setup_model(device)
    lossfn = setup_lossfn()

    train_dataset, val_dataset = setup_train_val_datasets(data_dir, dryrun=dryrun)
    train_dataset = copy.deepcopy(train_dataset)  # transformを設定した際にval_datasetに影響したくない
    train_transform = setup_crop_flip_transform()
    set_transform(train_dataset, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )

    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )
    n_iterations = len(train_loader) * n_epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)

    train2(
        model,
        lossfn,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        device=device,
    )

    loader, image_ids = setup_test_loader(data_dir, batch_size, dryrun)
    preds = predict(model, loader, device)
    write_prediction_with_clip(
        image_ids, preds, 0.0125, os.path.join(out_dir, "out.csv")
    )


if __name__ == "__main__":
    run()
