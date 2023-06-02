import time

import torch
from tqdm import tqdm

from model import SUASYOLO

USE_CUDA = False
device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

model = SUASYOLO(num_classes = 14).to(device)

# @profile
def test(num_repeats=10, batch_size=1):
    print(f"Testing batch size {batch_size}")
    start =  time.perf_counter()
    for _ in tqdm(range(num_repeats)):
        x = torch.rand((batch_size, 3, 640, 640))
        x=x.to(device)
        boxes, classes, objectness = model.predict(x)
    end = time.perf_counter()
    print(f"Time taken: {end-start} ({(end-start)/batch_size} per {num_repeats} images)")

@profile
def run_all_tests():
    test(batch_size=1)
    test(batch_size=2)
    test(batch_size=4)

run_all_tests()