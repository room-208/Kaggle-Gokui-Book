import argparse
import os

from torch import optim

from dataloader import setup_test_loader, setup_train_val_loaders
from device import setup_device
from loss import setup_lossfn
from model import setup_model
from predict import predict
from train import train
from writer import write_prediction


def train_subsec5(data_dir, batch_size, device, dryrun=False):
    model = setup_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lossfn = setup_lossfn()
    train_loader, val_loader = setup_train_val_loaders(data_dir, batch_size, dryrun)
    train(model, lossfn, optimizer, train_loader, val_loader, n_epochs=1, device=device)
    return model


def predict_subsec5(data_dir, out_dir, model, batch_size, device, dryrun=False):
    loader, image_ids = setup_test_loader(data_dir, batch_size, dryrun)
    preds = predict(model, loader, device)
    write_prediction(image_ids, preds, os.path.join(out_dir, "out.csv"))


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

    device = setup_device()
    model = train_subsec5(data_dir, batch_size, device, dryrun)
    predict_subsec5(data_dir, out_dir, model, batch_size, device, dryrun)


if __name__ == "__main__":
    run()
