import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from dataset import SUASDataset

from model import SUASYOLO
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchvision.ops import box_iou
from visualize import display_boxes

def eval_map_mar(model: SUASYOLO, dataset: SUASDataset, iou_threshold: float = 0.5, nms_threshold: float = 0.2, visualize=False):
    precisions = []
    recalls = []
    fig, axs = plt.subplots(1, len(dataset))
    for (img, label), ax in zip(dataset, axs):
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        boxes, objectness, classes = model.process_predictions(label.reshape(1, -1))
        boxes = boxes[objectness > 0].to("cpu")
        pred_boxes, pred_classes = model.predict(
            torch.tensor(img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0).to(DEVICE),
            nms_threshold=nms_threshold
        )
        pred_boxes = pred_boxes.to("cpu")
        if visualize:
            display_boxes(boxes, classes[objectness>0], (0,255,0),3,img)
            display_boxes(pred_boxes, pred_classes, (0,0,255),1,img)
            ax.imshow(img)
            ax.axis("off")
        ious = box_iou(boxes, pred_boxes)
        # filter by iou > threshold
        ious = ious[ious > iou_threshold]
        precision = len(ious) / len(pred_boxes)
        precisions.append(precision)
        if len(boxes)>0:
            recall = len(ious) / len(boxes)
            recalls.append(recall)
    toolbar = fig.canvas.toolbar
    toolbar.pack_forget()
    toolbar.update()
    plt.show()
    return np.mean(precisions), np.mean(recalls)

if __name__=='__main__':
    dataset = SUASDataset("data/images/tiny_train", "data/labels/tiny_train", 17, n_cells = 10)
    model = SUASYOLO(num_classes = 17, cell_resolution=10).to(DEVICE)
    model.load_state_dict(torch.load("tiny_train.pt"))
    model.eval()
    print(eval_map_mar(model, dataset, visualize=True))

