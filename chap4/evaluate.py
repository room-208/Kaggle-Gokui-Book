from pathlib import Path

import numpy as np
import torch
from angular_model import AngularModel
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from torchvision import transforms
from utils import extract_vectors, get_query_index_images


def evaluate(
    roxford5k_root_dir: str,
    outputs_dir: str,
    input_size: int = 256,
    device: str = "cuda",
):
    datasets = {
        "roxford5k": configdataset("roxford5k", roxford5k_root_dir),
        "rparis6k": configdataset("rparis6k", roxford5k_root_dir),
    }

    backbone = "resnet34"

    model = AngularModel(
        n_classes=3103,
        model_name=backbone,
        pretrained=True,
    )
    model.load_state_dict(
        torch.load(Path(outputs_dir, "arcface_last.pth"))["state_dict"],
    )
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # それぞれのデータセットで大域特徴を抽出して、中間ファイルに保存
    for dataset_name, dataset_config in datasets.items():
        index_images, query_images, bbxs = get_query_index_images(dataset_config)
        print(f"Extract index vectors on {dataset_name} ...")
        index_vectors = extract_vectors(
            model, index_images, input_size, 512, test_transform, device
        )
        print(f"Extract query vectors on {dataset_name} ...")
        query_vectors = extract_vectors(
            model, query_images, input_size, 512, test_transform, device, bbxs=bbxs
        )
        index_vectors = index_vectors.numpy()
        query_vectors = query_vectors.numpy()

        # 時間節約のため中間ファイルに保存
        np.save(
            f"{outputs_dir}/{dataset_name}_index.npy",
            index_vectors.astype(np.float32),
        )
        np.save(
            f"{outputs_dir}/{dataset_name}_query.npy",
            query_vectors.astype(np.float32),
        )

    # 大域特徴をロードして、内積に基づいて順位付けして評価
    for dataset_name, dataset_config in datasets.items():
        # shape = (n_dims, n_images)
        index_vectors = np.load(f"{outputs_dir}/{dataset_name}_index.npy")
        query_vectors = np.load(f"{outputs_dir}/{dataset_name}_query.npy")

        # shape = (n_index_images, n_query_images)
        scores = np.dot(index_vectors.T, query_vectors)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset_name, ranks, dataset_config["gnd"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--roxford5k_root_dir", required=True)
    parser.add_argument("--outputs_dir", required=True)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    roxford5k_root_dir = args.roxford5k_root_dir
    outputs_dir = args.outputs_dir
    device = args.device

    evaluate(
        roxford5k_root_dir=roxford5k_root_dir,
        outputs_dir=outputs_dir,
        input_size=256,
        device=device,
    )
