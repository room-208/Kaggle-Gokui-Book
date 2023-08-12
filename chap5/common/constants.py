from pathlib import Path

import joblib

IS_KAGGLE = True

if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/working/chap5-data")
    OUTPUT_DIR = Path("/kaggle/working/")
else:
    DATA_DIR = Path("../data")  # Path(os.getenv("QQP_DATA_DIR", "/data"))
    OUTPUT_DIR = Path("../outputs")

INPUT_DIR = DATA_DIR / "input"
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
