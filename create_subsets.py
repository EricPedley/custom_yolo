# for each folder (train, validation, test) in data/images and data/labels, create a subset of 10% of the data in adjacent folders named (train_10, validation_10, test_10) deterministically

import os
import shutil

percentage = 5

DATA_FOLDER = "data_v2"

for folder in ("train", "validation", "test"):
    # make folder
    # os.mkdir(f"{DATA_FOLDER}/images/{folder}_{percentage}")
    os.mkdir(f"{DATA_FOLDER}/labels/{folder}_{percentage}")
    n_files = len(os.listdir(f"{DATA_FOLDER}/images/{folder}"))
    n_files_after = int(n_files * percentage/100)
    for fname in os.listdir(f"{DATA_FOLDER}/images/{folder}")[:n_files_after]:
        # shutil.copy(f"{DATA_FOLDER}/images/{folder}/{fname}", f"{DATA_FOLDER}/images/{folder}_{percentage}/{fname}")
        numerical_part=  fname.split(".")[0][len("image"):]
        shutil.copy(f"{DATA_FOLDER}/labels/{folder}/image{numerical_part}.txt", f"{DATA_FOLDER}/labels/{folder}_{percentage}/image{numerical_part}.txt")

