import argparse

import numpy as np
import torch
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from device import setup_device
from model import setup_model
from transform import setup_transform


def extract_vectors(
    data_dir,
    model,
    image_files,
    input_size,
    out_dim,
    transform,
    device,
    bbxs=None,
):
    dataloader = DataLoader(
        ImagesFromList(
            root=data_dir,
            images=image_files,
            imsize=input_size,
            transform=transform,
            bbxs=bbxs,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    with torch.no_grad():
        vecs = torch.zeros(out_dim, len(image_files))
        for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.to(device)
            embed = model(x).squeeze()
            vecs[:, i] = embed
            # print(embed.shape)

    return vecs


def get_query_index_images(cfg, dryrun):
    index_images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
    query_images = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]
    bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]
    # try:
    #    bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]
    # except KeyError:
    #    bbxs = None

    if dryrun:
        index_images = index_images[:10]
        query_images = query_images[:10]
        bbxs = bbxs[:10]

    print("len(index_images)=", len(index_images))
    print("len(query_images)=", len(query_images))
    print("len(bbxs)=", len(bbxs))
    return index_images, query_images, bbxs


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
