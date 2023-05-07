import torch
import torch.nn as nn
from torchvision.ops import box_iou
import numpy as np
LAMBDA_NOOBJ = 0.5
LAMBDA_COORD = 5

class YoloLoss(nn.Module):
    def __init__(self, num_classes):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # targets shape is (s,s, 5+num_classes)
        # predictions shape is (s,s, 5*num_boxes+num_classes) but num_boxes is 1 for us so it's just the same
        # each vector is (x, y, w, h, objectness, class1, class2, ...)
        assert predictions.ndim == targets.ndim == 4
        assert predictions.shape[0] == targets.shape[0]
        predictions = predictions.reshape(-1, 5 + self.num_classes)
        targets = targets.reshape(-1, 5 + self.num_classes) 
        # object loss (whether or not there was an object in the tile)
        # normally you weigh the loss differently for false positives vs false negatives
        objectness_targets = targets[..., 4]
        contains_obj = (objectness_targets == 1).unsqueeze(1).expand_as(predictions)
        predictions_with_objs = contains_obj * predictions 
        targets_with_objs = contains_obj * targets 
        

        positive_object_loss = self.mse(predictions_with_objs[..., 4], objectness_targets)


        preds_without_objs = (~contains_obj) * predictions
        negative_object_loss = self.mse(preds_without_objs[..., 4], objectness_targets)

        object_loss = positive_object_loss + LAMBDA_NOOBJ * negative_object_loss
        
        # box loss
        box_coord_loss = self.mse(predictions_with_objs[..., :2], targets_with_objs[..., :2])# x and y loss
        box_size_loss = self.mse(torch.sign(predictions_with_objs[..., 2:4]) * torch.sqrt(torch.abs(predictions_with_objs[..., 2:4])), torch.sqrt(targets_with_objs[..., 2:4]))# w and h loss


        # class loss
        class_loss = self.mse(predictions_with_objs[..., 5:], targets_with_objs[..., 5:])

        # total loss
        total_loss = LAMBDA_COORD * (box_coord_loss + box_size_loss) + object_loss + class_loss
        return total_loss