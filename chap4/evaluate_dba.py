import faiss
import numpy as np
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print


def search_with_faiss_cpu(feat_test, feat_index, topk=5):
    feat_index = np.ascontiguousarray(feat_index, dtype=np.float32)
    feat_test = np.ascontiguousarray(feat_test, dtype=np.float32)

    n_dim = feat_index.shape[1]
    cpu_index = faiss.IndexFlatIP(n_dim)
    cpu_index.add(feat_index)

    dists, topk_idx = cpu_index.search(x=feat_test, k=topk)

    cpu_index.reset()
    del cpu_index

    return dists, topk_idx


def l2norm_numpy(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


def qe_dba(feats_test, feats_index, sims, topk_idx, alpha=3.0, qe=True, dba=True):
    # alpha-query expansion (DBA)
    feats_concat = np.concatenate([feats_test, feats_index], axis=0)
    weights = sims**alpha

    feats_tmp = np.zeros(feats_concat.shape)
    for i in range(feats_concat.shape[0]):
        feats_tmp[i, :] = weights[i].dot(feats_concat[topk_idx[i]])

    del feats_concat
    feats_concat = l2norm_numpy(feats_tmp).astype(np.float32)

    split_at = [len(feats_test)]
    if qe and dba:
        return np.split(feats_concat, split_at, axis=0)
    elif not qe and dba:
        _, feats_index = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    elif qe and not dba:
        feats_test, _ = np.split(feats_concat, split_at, axis=0)
        return feats_test, feats_index
    else:
        raise ValueError


def evaluate_dba(
    roxford5k_root_dir: str,
    outputs_dir: str,
):
    datasets = {
        "roxford5k": configdataset("roxford5k", roxford5k_root_dir),
        "rparis6k": configdataset("rparis6k", roxford5k_root_dir),
    }

    # 大域特徴をロードして、内積に基づいて順位付けして評価
    for dataset_name, dataset_config in datasets.items():
        # shape = (n_dims, n_images)
        index_vectors = np.load(f"{outputs_dir}/{dataset_name}_index.npy")
        query_vectors = np.load(f"{outputs_dir}/{dataset_name}_query.npy")

        # shape = (n_images, n_dims)
        index_vectors = index_vectors.T
        query_vectors = query_vectors.T
        index_vectors = l2norm_numpy(index_vectors)
        query_vectors = l2norm_numpy(query_vectors)

        # reranking
        n_qe, alpha = 3, 3.0
        feat_all = np.concatenate([query_vectors, index_vectors], axis=0)
        dists, topk_idx = search_with_faiss_cpu(feat_all, feat_all, topk=n_qe)
        query_vectors, index_vectors = qe_dba(
            query_vectors, index_vectors, dists, topk_idx, alpha=alpha
        )

        scores = np.dot(query_vectors, index_vectors.T)
        ranks = np.argsort(-scores, axis=1).T
        compute_map_and_print(dataset_name, ranks, dataset_config["gnd"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--roxford5k_root_dir", required=True)
    parser.add_argument("--outputs_dir", required=True)
    args = parser.parse_args()

    roxford5k_root_dir = args.roxford5k_root_dir
    outputs_dir = args.outputs_dir

    evaluate_dba(
        roxford5k_root_dir=roxford5k_root_dir,
        outputs_dir=outputs_dir,
    )
