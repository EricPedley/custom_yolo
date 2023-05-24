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
from tqdm import tqdm

def eval_map_mar(model: SUASYOLO, dataset: SUASDataset, conf_threshold: float = 0.5, iou_threshold: float = 0.5, visualize=False):
    precisions = []
    recalls = []
    if visualize:
        fig, axs = plt.subplots(1, len(dataset))
    else:
        axs = [None] * len(dataset)
    was_training = model.training
    model.eval()
    for (img, label), ax in zip(dataset, axs):
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        boxes, objectness, classes = model.process_predictions(label.reshape(1, -1))
        boxes = boxes[objectness > 0].to("cpu")
        pred_boxes, pred_classes, pred_objectness = model.predict(
            torch.tensor(img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0).to(DEVICE),
            conf_threshold=conf_threshold
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
        if len(pred_boxes)>0:
            precision = len(ious) / len(pred_boxes)
            precisions.append(precision)
        if len(boxes)>0:
            recall = len(ious) / len(boxes)
            recalls.append(recall)
    model.train(mode=was_training)
    if visualize:
        toolbar = fig.canvas.toolbar
        toolbar.pack_forget()
        toolbar.update()
        plt.show()
    return np.mean(precisions), np.mean(recalls)

def create_mAP_mAR_graph(model: SUASYOLO, test_dataset: SUASDataset, iou_threshold=0.5):
    mAPs = []
    mARs = []
    print("Calculating mAP vs mAR")
    for conf_threshold in tqdm(np.linspace(0, 1, 25)):
        mAP, mAR = eval_map_mar(model, test_dataset, conf_threshold=conf_threshold, iou_threshold=0.5)
        mAPs.append(mAP)
        mARs.append(mAR)
    print(list(zip(mAPs, mARs)))
    fig = plt.figure()
    plt.title(f"mAP vs mAR @ IOU={iou_threshold}")
    plt.xlabel("mAP")
    plt.ylabel("mAR")
    plt.plot(mAPs, mARs)
    return fig


if __name__=='__main__':
    dataset = SUASDataset("data/images/tiny_train", "data/labels/tiny_train", 17, n_cells = 10)
    model = SUASYOLO(num_classes = 17).to(DEVICE)
    model.load_state_dict(torch.load("tiny_train.pt"))
    model.eval()
    # print(eval_map_mar(model, dataset, visualize=True))
    fig = create_mAP_mAR_graph(model, dataset)
    fig.show()
    plt.show()

