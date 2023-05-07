from itertools import tee

import torch
import torch.nn as nn
from torchvision.ops import nms 

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x):
        return self.conv(x)

class SUASYOLO(nn.Module):
    def __init__(self, num_classes, cell_resolution = 7, img_size=(640, 640)):
        super(SUASYOLO, self).__init__()
        feature_depths = [
            3, 64, 128, 256, 512, 1024, 1024, 1024
        ]
        self.feature_extraction = nn.Sequential(*[
            ConvLayer(in_depth, out_depth, 3, 1, 1)
         for in_depth, out_depth, in pairwise(feature_depths)
        ])

        num_size_reductions = len(feature_depths) - 1

        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_depths[-1] * (img_size[0] // 2**num_size_reductions) * (img_size[1] // 2**num_size_reductions), 496),
            nn.LeakyReLU(0.1),
            nn.Linear(496, cell_resolution**2 * (5 + num_classes))
        )

        self.cell_resolution = cell_resolution
        self.num_classes = num_classes


    def forward(self, x) -> torch.Tensor:
        x = self.feature_extraction(x)
        return self.detector(x)

    def process_predictions(self, raw_predictions: torch.Tensor):
        assert raw_predictions.ndim == 2
        assert raw_predictions.shape[1] == self.cell_resolution**2 * (5 + self.num_classes)
        raw_predictions = raw_predictions.reshape(-1, self.cell_resolution, self.cell_resolution, 5 + self.num_classes)
        # each vector is (center_x, center_y, w, h, objectness, class1, class2, ...)
        # where the coordinates are a fraction of the cell size and relative to the top left corner of the cell
        boxes = raw_predictions[..., :4]
        objectness = raw_predictions[..., 4]
        classes = raw_predictions[..., 5:]

        boxes[..., :2] -= boxes[..., 2:] / 2 # adjust center coords to be top-left coords
        boxes*= 1/self.cell_resolution# scale to be percent of global image coords
        for i in range(boxes.shape[1]):# add offsets for each cell
            for j in range(boxes.shape[2]):
                boxes[..., i, j, 0] += j * (1/self.cell_resolution)
                boxes[..., i, j, 1] += i * (1/self.cell_resolution)
        boxes[..., 2:] += boxes[..., :2] # add width and height to get bottom-right coords

        boxes = boxes.reshape(-1, 4)
        objectness = objectness.reshape(-1)
        classes = classes.reshape(-1, self.num_classes)

        return boxes, objectness, classes

    def predict(self, x: torch.Tensor, nms_threshold=0.2, max_preds = 10):
        raw_predictions = self.forward(x)

        boxes, objectness, classes = self.process_predictions(raw_predictions)

        kept_indices = nms(boxes, objectness, nms_threshold) # todo: make this batched_nms and return the boxes per batch instead of as one

        return boxes[kept_indices][:max_preds], classes[kept_indices][:max_preds]