import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from eval import create_mAP_mAR_graph, eval_map_mar
from loss import FocalLoss
from dataset import SUASDataset
from model import SUASYOLO
from visualize import get_display_figures

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
IMG_DIR = "data/images/train"
LABEL_DIR = "data/labels/train"
IOU_THRESHOLD = 0.5 # iou threshold for nms
CONF_THRESHOLD = 0.5 # confidence threshold for calculating mAP and mAR

TENSORBOARD_LOGGING =True
if TENSORBOARD_LOGGING:
    num_prev_runs = len(os.listdir('runs')) 
    writer = SummaryWriter(f'runs/yolo-{num_prev_runs}')
def train_fn(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, dataloader: DataLoader, device: str, epochs: int, test_dataset):
    loop = tqdm(range(epochs), leave=True)
    for epoch_no in loop:
        mean_loss = []

        for batch_idx, (x, y) in enumerate(dataloader):
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

            if TENSORBOARD_LOGGING:
                step_no = epoch_no*len(dataloader) + batch_idx
                writer.add_scalar('Loss/train', loss_num, step_no)
                writer.add_scalar('Box Loss/train', box_loss.item(), step_no)
                writer.add_scalar('Object Loss/train', object_loss.item(), step_no) 
                writer.add_scalar('Class Loss/train', class_loss.item(), step_no)
                if epoch_no % 4 == 0 and batch_idx == 0:
                    mAP, mAR = eval_map_mar(model, test_dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)
                    writer.add_scalar('mAP/train', mAP, step_no)
                    writer.add_scalar('mAR/train', mAR, step_no) 




def main():
    model = SUASYOLO(num_classes = NUM_CLASSES).to(DEVICE)
    S = model.cell_resolution
    input_shape = (1, 3, 640, 640) 
    model_summary = summary(model, input_shape)
    if TENSORBOARD_LOGGING:
        writer.add_text("Model Summary", str(model_summary).replace('\n', '  \n'))
        writer.add_graph(model, torch.ones(input_shape).to(DEVICE))
    train_dataset = SUASDataset(IMG_DIR, LABEL_DIR, NUM_CLASSES, n_cells = S)
    test_dataset = SUASDataset("data/images/tiny_train", "data/labels/tiny_train", NUM_CLASSES, n_cells = S)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = FocalLoss(NUM_CLASSES)

    start = time.perf_counter()
    if TENSORBOARD_LOGGING:
        print(f"Starting training run {num_prev_runs}")
    train_fn(model, optimizer, loss_fn, train_loader, DEVICE, EPOCHS, test_dataset)
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
    # torch.save(model.state_dict(), "train.pt")


if __name__ == "__main__":
    main()