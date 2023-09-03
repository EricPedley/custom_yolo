import torch
import torch.nn as nn
from torchvision.ops import box_iou
import numpy as np
LAMBDA_NOOBJ = 0.5
LAMBDA_COORD = 5
LAMBDA_CLASS = 5
LAMBDA_COLOR = 5
from visualize import display_boxes
from model import SUASYOLO
import matplotlib.pyplot as plt

model = SUASYOLO(num_classes = 17)
class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction="sum")
        self.crossentropy = nn.CrossEntropyLoss(reduction="sum")
        self.bce = nn.BCELoss(reduction="sum")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        '''total_loss = box_loss + object_loss + class_loss'''

        # targets shape is (s,s, 5+num_classes)
        # predictions shape is (s,s, 5*num_boxes+num_classes) but num_boxes is 1 for us so it's just the same
        # each vector is (x, y, w, h, objectness, class1, class2, ...)
        assert predictions.ndim == targets.ndim == 4
        assert predictions.shape[0] == targets.shape[0]
        predictions = predictions.transpose(1, 3)
        targets = targets.transpose(1, 3)
        predictions = predictions.reshape(-1, 3 + 6 + self.num_classes + 36)
        targets = targets.reshape(-1, 3 + 6 + self.num_classes + 36) 
        # object loss (whether or not there was an object in the tile)
        # normally you weigh the loss differently for false positives vs false negatives
        objectness_targets = targets[..., 2]
        objectness_predictions = predictions[..., 2]
        contains_obj = objectness_targets == 1
        
        positive_object_loss = self.bce(objectness_predictions[contains_obj], objectness_targets[contains_obj])

        no_obj = ~contains_obj  
        negative_object_loss = self.bce(objectness_predictions[no_obj], objectness_targets[no_obj])

        if not contains_obj.any():
            return (torch.tensor(0), negative_object_loss, torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0))

        object_loss = positive_object_loss + LAMBDA_NOOBJ * negative_object_loss
        
        # box loss
        box_coord_loss = self.mse(predictions[..., :2][contains_obj], targets[..., :2][contains_obj])# x and y loss
        # box_size_loss = self.mse(torch.sign(predictions[..., 2:4][contains_obj]) * torch.sqrt(torch.abs(predictions[..., 2:4][contains_obj])), torch.sqrt(targets[..., 2:4][contains_obj]))# w and h loss
        box_loss = LAMBDA_COORD * (box_coord_loss)# + box_size_loss)

        # predictions[..., 5:] = torch.softmax(predictions[..., 5:], dim=1)

        # pt = predictions[..., 5:][contains_obj].where(targets[..., 5:][contains_obj] == 1, 1 - predictions[..., 5:][contains_obj])`
        
        # class_loss = -self.alpha*(1-pt)**self.gamma * torch.log(pt+1e-8)
        # class_loss = class_loss.mean() if class_loss.numel() > 0 else torch.tensor(0.0)

        # class loss
        shape_loss = LAMBDA_CLASS * self.mse(predictions[..., 9:9+self.num_classes][contains_obj], targets[..., 9:9+self.num_classes][contains_obj])

        non_person_object_indices = targets[contains_obj][..., 9:9+self.num_classes].argmax(dim=1) != 13 

        letter_loss = LAMBDA_CLASS * self.mse(predictions[..., 9+self.num_classes:][contains_obj][non_person_object_indices], targets[..., 9+self.num_classes:][contains_obj][non_person_object_indices])

        # color loss

        shape_color_loss = LAMBDA_COLOR * self.mse(predictions[..., 3:6][contains_obj][non_person_object_indices], targets[..., 3:6][contains_obj][non_person_object_indices])
        letter_color_loss = LAMBDA_COLOR * self.mse(predictions[..., 6:9][contains_obj][non_person_object_indices], targets[..., 6:9][contains_obj][non_person_object_indices])

        total_loss = box_loss + object_loss + shape_loss + letter_loss + shape_color_loss + letter_color_loss
        
        # total loss
        return (box_loss, object_loss, shape_loss, letter_loss, shape_color_loss, letter_color_loss) 
