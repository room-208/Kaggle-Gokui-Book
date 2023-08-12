import argparse
import os
from pathlib import Path

import joblib

DATA_DIR = Path("../data")  # Path(os.getenv("QQP_DATA_DIR", "/data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = Path("../outputs")
TRAIN_CSV_PATH = INPUT_DIR / "train.csv"
TEST_CSV_PATH = INPUT_DIR / "test.csv"
EMBEDDING_DIR = DATA_DIR / "embeddings"
GLOVE_PATH = EMBEDDING_DIR / "glove.840B.300d.bin"

FEATURE_MEMORY = joblib.Memory(DATA_DIR / "cache")

SPLIT_RANDOM_SEED = 1
EPS = 1e-10
NUM_PROCESSES = 4  # int(os.getenv("NUM_PROCESSES", 1))
NUM_TRAIN_SAMPLES = 404290
NUM_TEST_SAMPLES = 2345796
NUM_DRYRUN_SAMPLES = 1000


def set_constants_by_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_DIR", required=True, type=str)
    parser.add_argument("--OUTPUT_DIR", required=True, type=str)
    args = parser.parse_args()

    global DATA_DIR, OUTPUT_DIR, INPUT_DIR, TRAIN_CSV_PATH, TEST_CSV_PATH, EMBEDDING_DIR, GLOVE_PATH, FEATURE_MEMORY

    DATA_DIR = Path(args.DATA_DIR)
    OUTPUT_DIR = Path(args.OUTPUT_DIR)

    INPUT_DIR = DATA_DIR / "input"
    TRAIN_CSV_PATH = INPUT_DIR / "train.csv"
    TEST_CSV_PATH = INPUT_DIR / "test.csv"
    EMBEDDING_DIR = DATA_DIR / "embeddings"
    GLOVE_PATH = EMBEDDING_DIR / "glove.840B.300d.bin"
    FEATURE_MEMORY = joblib.Memory(DATA_DIR / "cache")
