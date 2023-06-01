# for each folder (train, validation, test) in data/images and data/labels, create a subset of 10% of the data in adjacent folders named (train_10, validation_10, test_10) deterministically

import os
import shutil

percentage = 5
for folder in ("train", "validation", "test"):
    # make folder
    os.mkdir(f"data/images/{folder}_{percentage}")
    os.mkdir(f"data/labels/{folder}_{percentage}")
    n_files = len(os.listdir(f"data/images/{folder}"))
    n_files_after = int(n_files * percentage/100)
    for fname in os.listdir(f"data/images/{folder}")[:n_files_after]:
        shutil.copy(f"data/images/{folder}/{fname}", f"data/images/{folder}_{percentage}/{fname}")
        numerical_part=  fname.split(".")[0][len("image"):]
        shutil.copy(f"data/labels/{folder}/image{numerical_part}.txt", f"data/labels/{folder}_{percentage}/image{numerical_part}.txt")

