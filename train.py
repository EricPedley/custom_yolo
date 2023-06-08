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

from eval import create_mAP_mAR_graph, eval_metrics
from loss import FocalLoss
from dataset import SUASDataset
from model import SUASYOLO
from visualize import get_display_figures

os.chdir(os.path.dirname(os.path.abspath(__file__)))

seed = 42
torch.manual_seed(seed)
LEARNING_RATE = 3e-4 # andrej karpathy magic number http://karpathy.github.io/2019/04/25/recipe/
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
WEIGHT_DECAY = 0
EPOCHS = 200 if os.getenv("EPOCHS") is None else int(os.getenv("EPOCHS"))
NUM_CLASSES = 14
NUM_WORKERS = 4
PIN_MEMORY = True
DATA_FOLDER = "data_v2"
TRAIN_DIRNAME = "train_100"
REDUCED_TRAIN_DIRNAME = "train_5"
VAL_DIRNAME = "validation_10"
TEST_DIRNAME = "test_100"
IOU_THRESHOLD = 0.50 # iou threshold for nms
CONF_THRESHOLD = 0.5 # confidence threshold for calculating mAP and mAR

TENSORBOARD_LOGGING = True 
if TENSORBOARD_LOGGING:
    num_prev_runs = len(os.listdir('runs')) 
    os.makedirs(f"weights/{num_prev_runs}")
    writer = SummaryWriter(f'runs/yolo-{num_prev_runs}')
def train_fn(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, dataloader: DataLoader, device: str, epochs: int, validation_dataset: SUASDataset, train_dataset: SUASDataset):
    loop = tqdm(range(epochs), leave=True)
    for epoch_no in loop:
        mean_loss = []
        if epoch_no%10 == 0:
            torch.save(model.state_dict(), f"weights/{num_prev_runs}/epoch_{epoch_no}.pt")

        for batch_idx, (x, y) in enumerate(dataloader):
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)
            out: torch.Tensor = model(x)
            box_loss, object_loss, shape_loss, letter_loss, shape_color_loss, letter_color_loss = loss_fn(out, y)
            loss = box_loss + object_loss + shape_loss + letter_loss + shape_color_loss + letter_color_loss
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
                writer.add_scalar('Shape Loss/train', shape_loss.item(), step_no)
                writer.add_scalar('Letter Loss/train', letter_loss.item(), step_no)
                if epoch_no % 5 == 0 and batch_idx == 0:
                    train_mAP, train_mAR, train_shapeconf, train_letterconf= eval_metrics(model, train_dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD, visualize=False)
                    val_mAP, val_mAR, val_shapeconf, val_letterconf = eval_metrics(model, validation_dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD, visualize=False)
                    # if mAP>0.9:
                    #     torch.save(model.state_dict(), f"overfit.pt")
                    #     break
                    writer.add_scalar('mAP/train', train_mAP, epoch_no)
                    writer.add_scalar('mAR/train', train_mAR, epoch_no) 
                    writer.add_scalar('Average Shape Ground-Truth Confidence/train', train_shapeconf, epoch_no)
                    writer.add_scalar('Average Letter Ground-Truth Confidence/train', train_letterconf, epoch_no)

                    writer.add_scalar('mAP/validation', val_mAP, epoch_no)
                    writer.add_scalar('mAR/validation', val_mAR, epoch_no) 
                    writer.add_scalar('Average Shape Ground-Truth Confidence/validation', val_shapeconf, epoch_no)
                    writer.add_scalar('Average Letter Ground-Truth Confidence/validation', val_letterconf, epoch_no)
                    writer.add_scalar('Shape Color Loss/train', shape_color_loss.item(), epoch_no)
                    writer.add_scalar('Letter Color Loss/train', letter_color_loss.item(), epoch_no)



def main():
    model = SUASYOLO(num_classes = NUM_CLASSES).to(DEVICE)
    S = model.num_cells
    input_shape = (1, 3, 640, 640) 
    model_summary = summary(model, input_shape)
    if TENSORBOARD_LOGGING:
        writer.add_text("Model Summary", str(model_summary).replace('\n', '  \n'))
        
        writer.add_graph(model, torch.ones(input_shape).to(DEVICE))
    train_dataset = SUASDataset(f"{DATA_FOLDER}/images/{TRAIN_DIRNAME.split('_')[0]}", f"{DATA_FOLDER}/labels/{TRAIN_DIRNAME}", NUM_CLASSES, n_cells = S)
    train_subset = SUASDataset(f"{DATA_FOLDER}/images/{REDUCED_TRAIN_DIRNAME.split('_')[0]}", f"{DATA_FOLDER}/labels/{REDUCED_TRAIN_DIRNAME}", NUM_CLASSES, n_cells = S)
    val_dataset = SUASDataset(f"{DATA_FOLDER}/images/{VAL_DIRNAME.split('_')[0]}", f"{DATA_FOLDER}/labels/{VAL_DIRNAME}", NUM_CLASSES, n_cells = S)
    test_dataset = SUASDataset(f"{DATA_FOLDER}/images/{TEST_DIRNAME.split('_')[0]}", f"{DATA_FOLDER}/labels/{TEST_DIRNAME}", NUM_CLASSES, n_cells = S)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = FocalLoss(NUM_CLASSES)

    start = time.perf_counter()
    if TENSORBOARD_LOGGING:
        print(f"Starting training run {num_prev_runs}")
    train_fn(model, optimizer, loss_fn, train_loader, DEVICE, EPOCHS, val_dataset, train_subset)
    end = time.perf_counter()
    print(f"Training took {end-start} seconds")
    model.eval()
    # create mAP vs mAR plot and write to tensorboard

    mAP_mAR_fig = create_mAP_mAR_graph(model, train_dataset) 
    visualizations = get_display_figures(model, train_dataset, n=min(5, BATCH_SIZE), centers_only=True)

    if TENSORBOARD_LOGGING:
        writer.add_figure("mAP vs mAR", mAP_mAR_fig)
        for i, fig in enumerate(visualizations):
            writer.add_figure(f"Visualization {i}", fig)

        # not sure why this doesn't work but would be nice to add once I get to hyperparameter tuning
        # mAP50, mAR50 = eval_metr(model, test_dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=0.5)
        # mAP75, mAR75 = eval_map_mar(model, test_dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=0.75)
        # mAP90, mAR90 = eval_map_mar(model, test_dataset, conf_threshold=CONF_THRESHOLD, iou_threshold=0.9)
        # writer.add_hparams(
        #     hparam_dict={
        #     "learning_rate": LEARNING_RATE,
        #     "batch_size": BATCH_SIZE,
        #     "weight_decay": WEIGHT_DECAY,
        #     "epochs": EPOCHS,
        #     "num_classes": NUM_CLASSES,
        #     "num_workers": NUM_WORKERS,
        #     "pin_memory": PIN_MEMORY,
        #     "iou_threshold": IOU_THRESHOLD,
        #     "validation_conf_threshold": CONF_THRESHOLD
        # }, metric_dict={
        #     "mAP@50": mAP50,
        #     "mAR@50": mAR50,
        #     "mAP@75": mAP75,
        #     "mAR@75": mAR75,
        #     "mAP@90": mAP90,
        #     "mAR@90": mAR90, 
        # })
    else:
        fig.show()
        plt.show()
    torch.save(model.state_dict(), f"weights/{num_prev_runs}/final.pt")


if __name__ == "__main__":
    main()