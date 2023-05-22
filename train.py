import torch
import torch.nn as nn
from torchvision.ops import box_iou, nms

from example.model import Yolov1 
#from example.loss import YoloLoss
from loss import YoloLoss
from dataset import SUASDataset

from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

# change directory to current file
import os
import time

import wandb


from model import SUASYOLO
os.chdir(os.path.dirname(os.path.abspath(__file__)))

seed = 42
torch.manual_seed(seed)
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_CLASSES = 17
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images/tiny_train"
LABEL_DIR = "data/labels/tiny_train"



def train_fn(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, dataloader: DataLoader, device: str):
    loop = tqdm(dataloader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x: torch.Tensor = x.to(device)
        y: torch.Tensor = y.to(device)
        out: torch.Tensor = model(x)
        loss: torch.Tensor = loss_fn(out.reshape_as(y), y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_num = loss.item()
        loop.set_postfix(loss=loss_num)
        wandb.log({"loss": loss_num})



def main():
    model = SUASYOLO(num_classes = NUM_CLASSES).to(DEVICE)
    S = model.cell_resolution
    model_summary = summary(model, (1, 3, 640, 640))
    print(model_summary)
    
    train_dataset = SUASDataset(IMG_DIR, LABEL_DIR, NUM_CLASSES, n_cells = S)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(NUM_CLASSES)
    wandb.init(
        # set the wandb project where this run will be logged
        project="custom-yolo",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LEARNING_RATE,
        "architecture": "YoloV1-Hybrid",
        "dataset": "UCI-SUAS-10",
        "epochs": 100,
        "Output Size (mb)":model_summary.to_megabytes(model_summary.total_output_bytes)
        }
    )
    start = time.perf_counter()
    try:
        for epoch in range(EPOCHS):
            train_fn(model, optimizer, loss_fn, train_loader, DEVICE)
    finally:
        end = time.perf_counter()

        print(f"Training took {end-start} seconds")
        torch.save(model.state_dict(), "tiny_train.pt")


if __name__ == "__main__":
    main()