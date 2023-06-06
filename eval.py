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

def eval_map_mar(model: SUASYOLO, dataset: SUASDataset, conf_threshold: float = 0.5, iou_threshold: float = 0.5, sampling_ratio=1.0, visualize=False):
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
        boxes, objectness, classes = model.process_predictions(label.unsqueeze(0))
        boxes = boxes[objectness > 0].to("cpu")
        pred_boxes, pred_classes, pred_objectness = model.predict(
            torch.tensor(img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0).to(DEVICE),
            conf_threshold=conf_threshold
        )
        pred_coords = pred_boxes.to("cpu")
        if visualize:
            display_boxes(boxes, classes[objectness>0], objectness, (0,255,0),3,img, centers_only=True)
            display_boxes(pred_boxes, pred_classes, pred_objectness, (0,0,255),1,img, centers_only=True)
            ax.imshow(img)
            ax.axis("off")
        # get number of boxes where the centers are closer than 50 * iou_threshold pixels
        distances_list = []
        for box in pred_boxes:
            x1, y1 = (box*640).to("cpu").type(torch.int).tolist()
            for box2 in boxes:
                x2, y2 = (box2*640).to("cpu").type(torch.int).tolist()
            distances_list.append(np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])))
        distances = np.array(distances_list) 
        true_positives = distances[distances < 50 * iou_threshold]

        # ious = box_iou(boxes, pred_boxes)
        # filter by iou > threshold
        # true_positives = ious[ious > iou_threshold]
        if len(pred_boxes)>0:
            precision = len(true_positives) / len(pred_boxes)
            precisions.append(precision)
        if len(boxes)>0:
            recall = len(true_positives) / len(boxes)
            recalls.append(recall)
    model.train(mode=was_training)
    if visualize:
        toolbar = fig.canvas.toolbar
        toolbar.pack_forget()
        toolbar.update()
        plt.show()
    # if np.mean(precisions)>0.9: raise Exception("I smell bullshit!")
    return np.mean(precisions) if len(precisions)>0 else 0, np.mean(recalls) if len(recalls)>0 else 0

def create_mAP_mAR_graph(model: SUASYOLO, test_dataset: SUASDataset, iou_threshold=0.5):
    mAPs = []
    mARs = []
    print("Calculating mAP vs mAR")
    for conf_threshold in tqdm(np.linspace(0, 1, 25)):
        mAP, mAR = eval_map_mar(model, test_dataset, conf_threshold=conf_threshold, iou_threshold=iou_threshold, visualize=False)
        mAPs.append(mAP)
        mARs.append(mAR)
    # print(list(zip(mAPs, mARs)))
    fig = plt.figure()
    plt.title(f"mAP vs mAR @ IOU={iou_threshold}")
    plt.xlabel("mAP")
    plt.ylabel("mAR")
    # set scales from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(mAPs, mARs)
    return fig


if __name__=='__main__':
    num_classes = 14
    model = SUASYOLO(num_classes = num_classes).to(DEVICE)
    data_folder = "test_10"
    dataset = SUASDataset(f"data/images/{data_folder}", f"data/labels/{data_folder}", num_classes, n_cells = model.num_cells)
    model.load_state_dict(torch.load("yolo_132.pt"))
    model.eval()
    print(eval_map_mar(model, dataset, visualize=False))
    fig = create_mAP_mAR_graph(model, dataset)
    fig.show()
    plt.show()

