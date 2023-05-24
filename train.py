import torch
import torch.nn as nn
from torchvision.ops import box_iou, nms, focal_loss
import numpy as np
import matplotlib.pyplot as plt
from example.model import Yolov1 
from eval import create_mAP_mAR_graph, eval_map_mar
#from example.loss import YoloLoss
from loss import FocalLoss
from dataset import SUASDataset

from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# change directory to current file
import os
import time

import wandb


from model import SUASYOLO
from visualize import get_display_figures
os.chdir(os.path.dirname(os.path.abspath(__file__)))

seed = 42
torch.manual_seed(seed)
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
WEIGHT_DECAY = 0
EPOCHS = 50 
NUM_CLASSES = 17
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images/tiny_train"
LABEL_DIR = "data/labels/tiny_train"
IOU_THRESHOLD = 0.5 # iou threshold for nms
CONF_THRESHOLD = 0.5 # confidence threshold for calculating mAP and mAR

WANDB_LOGGING=False
TENSORBOARD_LOGGING =True
if TENSORBOARD_LOGGING:
    num_prev_runs = len(os.listdir('runs')) 
    writer = SummaryWriter(f'runs/yolo-{num_prev_runs}')
def train_fn(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, dataloader: DataLoader, device: str, epochs: int):
    for epoch_no in range(epochs):
        print(f"Epoch {epoch_no+1}/{epochs}")
        loop = tqdm(dataloader, leave=True)
        mean_loss = []

        for batch_idx, (x, y) in enumerate(loop):
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)
            out: torch.Tensor = model(x)
            box_loss, object_loss, class_loss = loss_fn(out.reshape_as(y), y)
            loss = box_loss + object_loss + class_loss
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_num = loss.item()
            loop.set_postfix(loss=loss_num)
            mAP, mAR = eval_map_mar(model, dataloader.dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)
            if WANDB_LOGGING:
                wandb.log({
                    "loss": loss_num,
                    "box_loss": box_loss.item(),
                    "object_loss": object_loss.item(),
                    "class_loss": class_loss.item(),
                    "mAP": mAP,
                    "mAR": mAR
                    })
            if TENSORBOARD_LOGGING:
                writer.add_scalar('Loss/train', loss_num, epoch_no*len(dataloader) + batch_idx)
                writer.add_scalar('Box Loss/train', box_loss.item(), epoch_no*len(dataloader) + batch_idx)
                writer.add_scalar('Object Loss/train', object_loss.item(), epoch_no*len(dataloader) + batch_idx)
                writer.add_scalar('Class Loss/train', class_loss.item(), epoch_no*len(dataloader) + batch_idx)
                writer.add_scalar('mAP/train', mAP, epoch_no*len(dataloader) + batch_idx)
                writer.add_scalar('mAR/train', mAR, epoch_no*len(dataloader) + batch_idx)




def main():
    model = SUASYOLO(num_classes = NUM_CLASSES).to(DEVICE)
    S = model.cell_resolution
    input_shape = (1, 3, 640, 640) 
    model_summary = summary(model, input_shape)
    if TENSORBOARD_LOGGING:
        writer.add_text("Model Summary", str(model_summary))
        writer.add_graph(model, torch.ones(input_shape).to(DEVICE))
    train_dataset = SUASDataset(IMG_DIR, LABEL_DIR, NUM_CLASSES, n_cells = S)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = FocalLoss(NUM_CLASSES)
    if WANDB_LOGGING:
        wandb.init(
            # set the wandb project where this run will be logged
            project="custom-yolo",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": LEARNING_RATE,
                "architecture": "YoloV1-Hybrid",
                "dataset": "UCI-SUAS-10",
                "epochs": 100,
                "batch_size": BATCH_SIZE,
                "conf_threshold": CONF_THRESHOLD,
                "nms_threshold": IOU_THRESHOLD,
                "Output Size (mb)":model_summary.to_megabytes(model_summary.total_output_bytes)
            }
        )
    start = time.perf_counter()
    train_fn(model, optimizer, loss_fn, train_loader, DEVICE, EPOCHS)
    end = time.perf_counter()
    print(f"Training took {end-start} seconds")
    # create mAP vs mAR plot and write to tensorboard

    fig = create_mAP_mAR_graph(model, train_dataset) 
    visualizations = get_display_figures(model, train_dataset)

    if TENSORBOARD_LOGGING:
        writer.add_figure("mAP vs mAR", fig)
        for i, fig in enumerate(visualizations):
            writer.add_figure(f"Visualization {i}", fig)
    else:
        fig.show()
        plt.show()
    torch.save(model.state_dict(), "tiny_train.pt")


if __name__ == "__main__":
    main()