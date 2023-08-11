import argparse

import numpy as np
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from device import setup_device
from model import setup_model
from transform import setup_transform
from utils import extract_vectors, get_query_index_images


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--out_dim", default=512, type=int)
    parser.add_argument("--dryrun", action="store_true", default=False)
    args = parser.parse_args()

    root_dir = args.root_dir
    data_dir = args.data_dir
    input_size = args.input_size
    out_dim = args.out_dim
    dryrun = args.dryrun

    device = setup_device()

    detasets = {
        "roxford5k": configdataset("roxford5k", data_dir),
        "rparis6k": configdataset("rparis6k", data_dir),
    }

    model = setup_model(device)
    model.eval()

    transform = setup_transform(input_size)

    for dataset_name, cfg in detasets.items():
        index_images, query_images, bbxs = get_query_index_images(cfg, dryrun)
        index_vectors = extract_vectors(
            root_dir, model, index_images, input_size, out_dim, transform, device
        )
        query_vectors = extract_vectors(
            root_dir, model, query_images, input_size, out_dim, transform, device
        )
        index_vectors = index_vectors.numpy()
        query_vectors = query_vectors.numpy()

        scores = np.dot(index_vectors.T, query_vectors)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset_name, ranks, cfg["gnd"])


if __name__ == "__main__":
    run()
