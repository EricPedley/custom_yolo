import math
from itertools import tee

import torch
import torch.nn as nn
from torchinfo import summary
from torchvision.ops import batched_nms 
import numpy as np

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def flatten(l):
  out = []
  for item in l:
    if isinstance(item, (list, tuple)):
      out.extend(flatten(item))
    else:
      out.append(item)
  return out

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DWConv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
        #     nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
        # )
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            groups=math.gcd(in_channels, out_channels), 
            bias=bias
        )
       
    def forward(self, x):
        return self.conv(x)
    
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, hidden_channels=None):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_channels is None:
            hidden_channels = out_channels//2
        self.shrink = ConvLayer(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.expand = ConvLayer(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
            # set shortcut conv weights to identity
            self.shortcut_conv.weight.data.fill_(0)
            for i in range(out_channels):
                self.shortcut_conv.weight.data[i, i%in_channels, 0, 0] = 1
            self.shortcut_conv.weight.requires_grad = False
        self.silu = nn.SiLU()
    def forward(self, x):
        skip_connection = x if self.in_channels == self.out_channels else self.shortcut_conv(x)
        return self.silu(skip_connection+self.expand(self.shrink(x)))

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.conv(x)
    
class SUASYOLO(nn.Module):
    def __init__(self, num_classes, img_width=640):
        super(SUASYOLO, self).__init__()
        self.num_classes = num_classes
        self.feature_extraction = nn.Sequential(
            ConvLayer(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvLayer(192, 128, 1, 1, 0),
            ConvLayer(128, 256, 3, 1, 1),
            ConvLayer(256, 256, 1, 1, 0),
            ConvLayer(256, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *flatten([
                [ConvLayer(512, 256, 1, 1, 0), ConvLayer(256, 512, 3, 1, 1)] for _ in range(4)
            ]),
            ConvLayer(512, 512, 1, 1, 0),
            ConvLayer(512, 1024, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *flatten([
                [ConvLayer(1024, 512, 1, 1, 0), ConvLayer(512, 1024, 3, 1, 1)] for _ in range(2)
            ]),
            ConvLayer(1024, 1024, 3, 1, 1),
            ConvLayer(1024, 1024, 3, 2, 1),
            ConvLayer(1024, 1024, 3, 1, 1),
        )
        self.num_cells = img_width // 64 
        S = self.num_cells
        C = self.num_classes
        hidden_size = 512
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S*S, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, S * S * (C + 5)),
            # nn.Conv2d(feature_depths[-1], hidden_size, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(hidden_size, 5+num_classes, kernel_size=3, stride=1, padding=1),
        )



    def forward(self, x) -> torch.Tensor:
        # visualize x
        # if x.shape[0]>1:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(x[0][0].detach().cpu().numpy().astype(np.uint8))
        #     plt.show()

        x = self.feature_extraction(x)
        x = self.detector(x)
        x = x.reshape(-1, (self.num_classes+5), self.num_cells, self.num_cells)
        # x[:,4,:,:] = torch.sigmoid(x[:,4,:,:]) # objectness (empirically, applying the sigmoid here actually makes the mAP slightly  worse)
        # x[:,5:,:,:] = torch.softmax(x[:,5:,:,:], dim=1) # class predictions
        # x[:,5:,:,:] = torch.sigmoid(x[:,5:,:,:]) # class predictions
        # x = nn.Flatten()(x)
        return x

    def process_predictions(self, raw_predictions: torch.Tensor):
        '''Returns (boxes, objectness, classes)'''
        # each vector is (center_x, center_y, w, h, objectness, class1, class2, ...)
        # where the coordinates are a fraction of the cell size and relative to the top left corner of the cell
        raw_predictions = torch.transpose(raw_predictions, 1, 3)
        boxes = raw_predictions[..., :4]
        objectness = raw_predictions[..., 4]
        classes = raw_predictions[..., 5:]

        boxes[..., :2] -= boxes[..., 2:] / 2 # adjust center coords to be top-left coords
        boxes*= 1/self.num_cells# scale to be percent of global image coords
        for i in range(boxes.shape[1]):# add offsets for each cell
            for j in range(boxes.shape[2]):
                boxes[..., i, j, 0] += j * (1/self.num_cells)
                boxes[..., i, j, 1] += i * (1/self.num_cells)
        boxes[..., 2:] += boxes[..., :2] # add width and height to get bottom-right coords

        boxes = boxes.reshape(-1, 4)
        objectness = objectness.reshape(-1)
        classes = classes.reshape(-1, self.num_classes)

        return boxes, objectness, classes

    def predict(self, x: torch.Tensor, conf_threshold = 0.5, iou_threshold=0.5, max_preds = 10) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        '''
        Returns (boxes, classes, objectness) where each is a tensor of shape (n, 4), (n, num_classes), (n, 1)
        '''
        n_batches = x.shape[0]
        raw_predictions = self.forward(x)

        boxes, objectness, classes = self.process_predictions(raw_predictions)
        batch_indices = torch.arange(n_batches).repeat_interleave(boxes.shape[0]//n_batches).to(x.device)
        batch_indices = batch_indices[objectness > conf_threshold]
        boxes = boxes[objectness > conf_threshold]
        classes = classes[objectness > conf_threshold]
        objectness = objectness[objectness > conf_threshold]

        kept_indices = batched_nms(boxes, objectness, batch_indices,iou_threshold) # todo: make this batched_nms and return the boxes per batch instead of as one

        return boxes[kept_indices][:max_preds], classes[kept_indices][:max_preds], objectness[kept_indices][:max_preds]
    
if __name__=="__main__":
    model = SUASYOLO(num_classes=14, img_width=640)
    input_shape = (1,3,640,640)
    summary(model, input_shape)