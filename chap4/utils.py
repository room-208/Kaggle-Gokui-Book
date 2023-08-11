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
            try:
                embed = model(x).squeeze()
            except TypeError:
                embed = model.extract_features(x)
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
