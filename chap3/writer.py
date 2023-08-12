import numpy as np


def write_prediction(image_ids, prediction, out_path):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            f.write(f"{i},{p}\n")


def write_prediction_with_clip(image_ids, prediction, clip_th, out_path):
    with open(out_path, "w") as f:
        f.write("id,label\n")
        for i, p in zip(image_ids, prediction):
            p = np.clip(p, clip_th, 1.0 - clip_th)
            f.write(f"{i},{p}\n")
