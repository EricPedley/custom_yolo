import torch
import torch.nn as nn
from torchvision.ops import box_iou, nms

from example.model import Yolov1 
#from example.loss import YoloLoss
from loss import YoloLoss
from dataset import SUASDataset

from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

# change directory to current file
import os
import time

from model import SUASYOLO
os.chdir(os.path.dirname(os.path.abspath(__file__)))

seed = 42
torch.manual_seed(seed)
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
WEIGHT_DECAY = 0
EPOCHS = 5
NUM_CLASSES = 17
NUM_WORKERS = 2
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
        out: torch.Tensor = model(x.permute(0, 3, 1, 2).type(torch.float32))
        loss: torch.Tensor = loss_fn(out.reshape_as(y), y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def main():
    S =10 
    #model = SUASYOLO(num_classes = NUM_CLASSES, cell_resolution=S).to(DEVICE)
    model = Yolov1(split_size=S, num_boxes=1, num_classes=NUM_CLASSES).to(DEVICE)
    print(summary(model, (3, 640, 640)))
    train_dataset = SUASDataset(IMG_DIR, LABEL_DIR, NUM_CLASSES, n_cells = S)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(NUM_CLASSES)
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        train_fn(model, optimizer, loss_fn, train_loader, DEVICE)
    end = time.perf_counter()
    print(f"Training took {end-start} seconds")


if __name__ == "__main__":
    main()