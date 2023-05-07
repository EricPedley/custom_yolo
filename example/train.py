"""
Main file for training Yolo model on Pascal VOC dataset

"""

import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from torchsummary import summary
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from model2 import SUASYOLO
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
from loss2 import YoloLoss as YoloLoss2
import os
seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 5 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 5 
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "../data/images/tiny_train"
LABEL_DIR = "../data/labels/tiny_train"

# change directory to file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


# transform = Compose([transforms.Resize((448, 448))]) 
transform = None#Compose([transforms.Resize((448, 448))]) 

def train_fn(train_loader, model, optimizer, loss_fn, epoch_no):
    loop = tqdm(train_loader, desc="Batch", position=1, leave=False)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out.reshape(-1, 10, 10, 22), y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    tqdm.write(f"Mean loss epoch {epoch_no} was {sum(mean_loss)/len(mean_loss)}")


def main():
    S = 10 
    C = 17
    B = 1
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)
    #model = SUASYOLO(num_classes=C, cell_resolution=S).to(DEVICE)
    print(summary(model, (3, S*64, S*64)))
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=S, C=C, B=1)
    #loss_fn = YoloLoss2(C)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S = S,
        C = C,
        B = B
    )

    test_dataset = VOCDataset(
        transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR, S=S, C=C, B=B
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    
    start = time.perf_counter()
    for epoch in tqdm(range(EPOCHS), desc="Epoch", position=0, leave=False):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        # pred_boxes, target_boxes = get_bboxes(
        #     train_loader, model, iou_threshold=0.5, threshold=0.4, S=S, C=17
        # )

        # mean_avg_prec = mean_average_precision(
        #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        # )
        # tqdm.write(f"Train mAP: {mean_avg_prec}")

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        train_fn(train_loader, model, optimizer, loss_fn, epoch)
    end = time.perf_counter()
    print(f"Training took {end - start}s")

if __name__ == "__main__":
    main()
