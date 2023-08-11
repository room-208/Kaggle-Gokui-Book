from pathlib import Path
from typing import Any

import torch
from cirtorch.datasets.genericdataset import ImagesFromList
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_checkpoint(model: Any, epoch: int, path: Path):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
        },
        path,
    )


def extract_vectors(
    model,
    image_files,
    input_size,
    out_dim,
    transform,
    device_str,
    bbxs=None,
    print_freq=1000,
):
    dataloader = torch.utils.data.DataLoader(
        ImagesFromList(
            root="",
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
        for i, X in enumerate(dataloader):
            if i % print_freq == 0:
                print(f"Processing {i} of {len(dataloader.dataset)}")
            X = X.to(device_str)
            vecs[:, i] = model.extract_features(X)
    return vecs


def get_query_index_images(cfg):
    index_images = [cfg["im_fname"](cfg, i) for i in range(cfg["n"])]
    query_images = [cfg["qim_fname"](cfg, i) for i in range(cfg["nq"])]

    try:
        bbxs = [tuple(cfg["gnd"][i]["bbx"]) for i in range(cfg["nq"])]
    except KeyError:
        bbxs = None

    return index_images, query_images, bbxs
