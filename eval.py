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
from loss import FocalLoss

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
    shape_confidences = []
    letter_confidences = []
    box_losses = []
    object_losses = []
    shape_losses = []
    letter_losses = []
    shape_color_losses = []
    letter_color_losses = []
    loss_fn = FocalLoss(14)

    for (img, label), ax in zip(dataset, axs):
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        boxes, objectness, shape_colors, letter_colors, shape_classes, letter_classes = model.process_predictions(label.unsqueeze(0))
        boxes = boxes[objectness>0]
        shape_classes = shape_classes[objectness>0]
        letter_classes = letter_classes[objectness>0]
        shape_colors = shape_colors[objectness>0]
        letter_colors = letter_colors[objectness>0]
        objectness = objectness[objectness>0]
        model_input = torch.tensor(img).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0).to(DEVICE)
        pred_boxes, pred_objectness, pred_shape_colors, pred_letter_colors, pred_shape_classes, pred_letter_classes, raw_predictions = model.predict(
            model_input,
            conf_threshold=conf_threshold
        )
        
        box_loss, object_loss, shape_loss, letter_loss, shape_color_loss, letter_color_loss = loss_fn(raw_predictions.to("cpu"), label.unsqueeze(0))

        box_losses.append(box_loss.item())
        object_losses.append(object_loss.item())
        shape_losses.append(shape_loss.item())
        letter_losses.append(letter_loss.item())
        shape_color_losses.append(shape_color_loss.item())
        letter_color_losses.append(letter_color_loss.item())
        
        if visualize:
            display_boxes(boxes, shape_classes, objectness, (0,255,0),3,img, centers_only=True)
            display_boxes(pred_boxes, pred_shape_classes, pred_objectness, (0,0,255),1,img, centers_only=True)
            ax.imshow(img)
            ax.axis("off")
        # get number of boxes where the centers are closer than 50 * iou_threshold pixels
        distances_list = []
        shape_local_conf = []
        letter_local_conf = []
        for pred_box, pred_shape_probs, pred_letter_probs in zip(pred_boxes, pred_shape_classes, pred_letter_classes):
            pred_x, pred_y = (pred_box*640).type(torch.int).tolist()
            for box2, shape_probs, letter_probs in zip(boxes, shape_classes, letter_classes):
                if shape_probs.sum() == 0:
                     continue
                real_shape_class = torch.argmax(shape_probs)
                real_letter_class = torch.argmax(letter_probs)

                shape_local_conf.append(pred_shape_probs[real_shape_class])
                letter_local_conf.append(pred_letter_probs[real_letter_class])

                x2, y2 = (box2*640).type(torch.int).tolist()
                distances_list.append(np.linalg.norm(np.array([pred_x, pred_y]) - np.array([x2, y2])))
        distances = np.array(distances_list) 
        shape_local_conf = np.array(shape_local_conf)
        letter_local_conf = np.array(letter_local_conf)

        valid_mask = distances < 50 * iou_threshold

        shape_local_conf = shape_local_conf[valid_mask]
        letter_local_conf = letter_local_conf[valid_mask]

        if valid_mask.sum() > 0:
            shape_confidences.append(np.mean(shape_local_conf))
            letter_confidences.append(np.mean(letter_local_conf))

        true_positive_distances = distances[valid_mask]
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

    validation_loss = (
        np.mean(box_losses),
        np.mean(object_losses),
        np.mean(shape_losses),
        np.mean(letter_losses),
        np.mean(shape_color_losses),
        np.mean(letter_color_losses)
    )
    
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
        np.mean(shape_confidences) if len(shape_confidences)>0 else 0,
        np.mean(letter_confidences) if len(letter_confidences)>0 else 0,
        validation_loss
    )

def create_mAP_mAR_graph(model: SUASYOLO, test_dataset: SUASDataset, iou_threshold=0.5):
    mAPs = []
    mARs = []
    print("Calculating mAP vs mAR")
    for conf_threshold in tqdm(np.linspace(0, 1, 25)):
        mAP, mAR, *_other_metrics = eval_metrics(model, test_dataset, conf_threshold=conf_threshold, iou_threshold=iou_threshold, visualize=False)
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
    split_folder = "test_100"
    data_folder = "data_v2"
    dataset = SUASDataset(f"{data_folder}/images/{split_folder.split('_')[0]}", f"{data_folder}/labels/{split_folder}", num_classes, n_cells = model.num_cells)
    model.load_state_dict(torch.load("weights/215/final.pt"))
    model.eval()
    mAP, mAR, shape_conf, letter_conf, losses = eval_metrics(model, dataset, conf_threshold=0.5, iou_threshold=0.5, visualize=False)
    print(f"mAP: {mAP}\nmAR: {mAR}\nshape_conf: {shape_conf}\nletter_conf: {letter_conf}")
    print(f"box_loss: {losses[0]}\nobject_loss: {losses[1]}\nshape_loss: {losses[2]}\nletter_loss: {losses[3]}\nshape_color_loss: {losses[4]}\nletter_color_loss: {losses[5]}")
    # fig = create_mAP_mAR_graph(model, dataset)
    # fig.show()
    # plt.show()

