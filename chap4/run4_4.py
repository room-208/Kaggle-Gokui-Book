import argparse

from evaluate import evaluate
from train import train


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--path_train_csv", required=True)
    parser.add_argument("--path_val_csv", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--gldv2_micro_path", required=True)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--dryrun", action="store_true", default=False)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--outputs_dir", required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    path_train_csv = args.path_train_csv
    path_val_csv = args.path_val_csv
    checkpoint_path = args.checkpoint_path
    gldv2_micro_path = args.gldv2_micro_path
    device = args.device
    dryrun = args.dryrun
    batch_size = args.batch_size
    outputs_dir = args.outputs_dir

    train(
        path_train_csv,
        path_val_csv,
        checkpoint_path,
        gldv2_micro_path,
        dryrun=dryrun,
        batch_size=batch_size,
        device=device,
    )
    evaluate(
        data_dir,
        outputs_dir,
        checkpoint_path,
        dryrun=dryrun,
        device=device,
    )


if __name__ == "__main__":
    run()
