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

def eval_metrics(model: SUASYOLO, dataset: SUASDataset, conf_threshold: float = 0.5, iou_threshold: float = 0.5, visualize=False):
    '''
    Returns a tuple (mAP, mAR, top-1 accuracy, top-5 accuracy, mean ground truth class confidence)
    '''
    precisions = []
    recalls = []
    if visualize:
        fig, axs = plt.subplots(1, len(dataset))
    else:
        axs = [None] * len(dataset)
    was_training = model.training
    model.eval()
    model.requires_grad_(False)
    top1_scores = []
    top5_scores = []
    class_confidences = []
    for (img, label), ax in zip(dataset, axs):
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        boxes, objectness, _shape_colors, _letter_colors, classes, _letter_classes = model.process_predictions(label.unsqueeze(0))
        boxes = boxes[objectness>0]
        classes = classes[objectness>0]
        pred_boxes, pred_classes, pred_objectness = model.predict(
            torch.tensor(img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0).to(DEVICE),
            conf_threshold=conf_threshold
        )
        # put that shit on the cpu
        pred_boxes = pred_boxes.to("cpu")
        pred_classes = pred_classes.to("cpu")

        
        if visualize:
            display_boxes(boxes, classes, objectness, (0,255,0),3,img, centers_only=True)
            display_boxes(pred_boxes, pred_classes, pred_objectness, (0,0,255),1,img, centers_only=True)
            ax.imshow(img)
            ax.axis("off")
        # get number of boxes where the centers are closer than 50 * iou_threshold pixels
        distances_list = []
        top1_scores_locally = []
        top5_scores_locally = [] 
        candidate_class_confidences = []
        for box, class_probs_1 in zip(pred_boxes, pred_classes):
            x1, y1 = (box*640).type(torch.int).tolist()
            for box2, class_probs_2 in zip(boxes, classes):
                ground_truth_class = torch.argmax(class_probs_2)
                top5_classes = torch.argsort(class_probs_1, descending=True)[:5]
                candidate_class_confidences.append(class_probs_1[ground_truth_class])
                top1_scores_locally.append(ground_truth_class == top5_classes[0])
                top5_scores_locally.append(ground_truth_class in top5_classes)
                x2, y2 = (box2*640).type(torch.int).tolist()
                distances_list.append(np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])))
        distances = np.array(distances_list) 
        top1_scores_locally = np.array(top1_scores_locally).astype(np.uint8)
        top5_scores_locally = np.array(top5_scores_locally).astype(np.uint8)
        candidate_class_confidences = np.array(candidate_class_confidences)

        true_top1_scores = top1_scores_locally[distances < 50 * iou_threshold]
        true_top5_scores = top5_scores_locally[distances < 50 * iou_threshold]
        candidate_class_confidences = candidate_class_confidences[distances < 50 * iou_threshold]

        if len(true_top1_scores) > 0:
            top1_scores.append(np.mean(true_top1_scores))
            top5_scores.append(np.mean(true_top5_scores))
            class_confidences.append(np.mean(candidate_class_confidences))

        true_positive_distances = distances[distances < 50 * iou_threshold]
        num_true_positives = len(true_positive_distances)

        # ious = box_iou(boxes, pred_boxes)
        # filter by iou > threshold
        # true_positives = ious[ious > iou_threshold]
        if len(pred_boxes)>0:
            precision = num_true_positives / len(pred_boxes)
            precisions.append(precision)
        if len(boxes)>0:
            recall = num_true_positives / len(boxes)
            recalls.append(recall)
    if was_training:
        model.train()
        model.requires_grad_(True)
    if visualize:
        toolbar = fig.canvas.toolbar
        toolbar.pack_forget()
        toolbar.update()
        plt.show()
    # if np.mean(precisions)>0.9: raise Exception("I smell bullshit!")
    return (
        np.mean(precisions) if len(precisions)>0 else 0, 
        np.mean(recalls) if len(recalls)>0 else 0,
        np.mean(top1_scores) if len(top1_scores)>0 else 0,
        np.mean(top5_scores) if len(top5_scores)>0 else 0,
        np.mean(class_confidences) if len(class_confidences)>0 else 0
    )

def create_mAP_mAR_graph(model: SUASYOLO, test_dataset: SUASDataset, iou_threshold=0.5):
    mAPs = []
    mARs = []
    print("Calculating mAP vs mAR")
    for conf_threshold in tqdm(np.linspace(0, 1, 25)):
        mAP, mAR, _top1, _top5, _conf = eval_metrics(model, test_dataset, conf_threshold=conf_threshold, iou_threshold=iou_threshold, visualize=False)
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
    dataset = SUASDataset(f"data/images/{data_folder.split('_')[0]}", f"data/labels/{data_folder}", num_classes, n_cells = model.num_cells)
    model.load_state_dict(torch.load("weights/good_181_coordsonly/final.pt"))
    model.eval()
    print(eval_metrics(model, dataset, visualize=False))
    # fig = create_mAP_mAR_graph(model, dataset)
    # fig.show()
    # plt.show()

