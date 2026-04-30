import os
import torch

IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:

    BASE_DATA_PATH = "/kaggle/input/coco-2017-dataset/coco2017"
    TRAIN_PATH = os.path.join(BASE_DATA_PATH, "train2017")
    VAL_PATH = os.path.join(BASE_DATA_PATH, "val2017")
    OUTPUT_DIR = "/kaggle/working"
else:
    TRAIN_PATH = "./data/train_samples"
    VAL_PATH = "./data/val_samples"
    OUTPUT_DIR = "./outputs"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999
EPOCHS = 20

L1_LAMBDA = 100.0