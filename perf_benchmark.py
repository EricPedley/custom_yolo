import time

import torch

from model import SUASYOLO

model = SUASYOLO(num_classes = 17).to("cuda")

# @profile
def test(num_repeats=100, batch_size=1):
    start =  time.perf_counter()
    for _ in range(num_repeats):
        x = torch.rand((batch_size, 3, 640, 640))
        x=x.to("cuda")
        boxes, classes, objectness = model.predict(x)
    end = time.perf_counter()
    print(f"Time taken: {end-start}")

print("Single batch")
test(num_repeats=100, batch_size=4)
# print("Batch size 4")
# test(num_repeats=100, batch_size=4)
# print("Batch size 8")
# test(num_repeats=100, batch_size=8)