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

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DWConv(nn.Module):
    '''
    Drop-in replacement for nn.Conv2d
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
        )
        # self.conv = nn.Conv2d(
        #     in_channels, 
        #     out_channels, 
        #     kernel_size, 
        #     stride, 
        #     padding, 
        #     groups=math.gcd(in_channels, out_channels), 
        #     bias=bias
        # )
        self.stride=stride # for calculating how many size reductions there are later
       
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
    
class ResBlock(nn.Module):
    def __init__(self, channels, num_repeats):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(channels, channels//2, kernel_size=1, stride=1, padding=0, use_depthwise=True),
                    ConvLayer(channels//2, channels, kernel_size=3, stride=1, padding=1, use_depthwise=True)
                )
                for _ in range(num_repeats)
            ]
        )
    def forward(self, x):
        for conv in self.convs:
            x = conv(x) + x
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, use_depthwise = False):
        super(ConvLayer, self).__init__()
        conv_class = DWConv if use_depthwise else nn.Conv2d
        self.conv = nn.Sequential(
            conv_class(in_channels, out_channels, kernel_size, stride, padding, bias=bias), 
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
            ConvLayer(3, 16, kernel_size=3, stride=1, padding=1), 
            ConvLayer(16, 32, kernel_size=3, stride=2, padding=1),
            *[C2f(32, 32) for _ in range(1)],
            ConvLayer(32, 64, kernel_size=3, stride=2, padding=1),
            *[C2f(64, 64) for _ in range(2)],
            ConvLayer(64, 128, kernel_size=3, stride=2, padding=1),
            *[C2f(128, 128) for _ in range(2)],
            ConvLayer(128, 256, kernel_size=3, stride=2, padding=1),
            *[C2f(256, 256) for _ in range(1)],
            SPPF(256, 256, 5)
        )
        # automatically calculate number of stride 2 convs in feature extraction
        num_size_reductions = 0
        for layer in self.feature_extraction:
            if isinstance(layer, ConvLayer) and layer.conv[0].stride == (2, 2):
                num_size_reductions += 1
        S = img_width // (2**num_size_reductions) # 640 // 32 = 20 
        self.num_cells = S 
        C = self.num_classes
        hidden_size=512
        self.detector = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(1024 * S*S, hidden_size),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, (self.num_cells ** 2) * (3 + 6 + C + 36)),
            ConvLayer(256, 256, kernel_size=3, stride=1, padding=1),
            ConvLayer(256, 256, kernel_size=3, stride=1, padding=1),
            ConvLayer(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, (3+6+C+36), kernel_size=1, stride=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Sequential(
            self.sigmoid,
            nn.Softmax(dim=1)
        )



    def forward(self, x) -> torch.Tensor:
        # visualize x
        # if x.shape[0]>1:
        #     import matplotlib.pyplot as plt
        #     plt.imshow(x[0][0].detach().cpu().numpy().astype(np.uint8))
        #     plt.show()

        x = self.feature_extraction(x)
        x = self.detector(x)
        x = x.reshape(-1, (3 + 6 + self.num_classes + 36), self.num_cells, self.num_cells)
        x[:, :2, :, :] = self.sigmoid(x[:, :2, :, :]) # box offset (doesn't include dimensions) 
        x[:,2,:,:] = self.sigmoid(x[:,2,:,:]) # objectness (empirically, applying the sigmoid here actually makes the mAP slightly  worse)
        x[:,3:6,:,:] = self.sigmoid(x[:,3:6,:,:]) # shape color
        x[:,6:9,:,:] = self.sigmoid(x[:,6:9,:,:]) # letter color

        #x[:,9:9+self.num_classes,:,:] = self.sigmoid(x[:,9:9+self.num_classes,:,:]) # shape class predictions
        #x[:, 9+self.num_classes:, :, :] = self.sigmoid(x[:, 9+self.num_classes:, :, :]) # letter class predictions

        # x[:,5:,:,:] = self.sigmoid(x[:,5:,:,:]) # class predictions
        # x[:,5:,:,:] = self.softmax(x[:,5:,:,:]) # class predictions
        # x[:,5:,:,:] = torch.sigmoid(x[:,5:,:,:]) # class predictions
        # x = nn.Flatten()(x)
        return x

    def process_predictions(self, raw_predictions: torch.Tensor):
        '''Returns (boxes, objectness, classes)'''
        # each vector is (center_x, center_y, w, h, objectness, class1, class2, ...)
        # where the coordinates are a fraction of the cell size and relative to the top left corner of the cell
        raw_predictions = torch.transpose(raw_predictions, 1, 3)
        boxes = raw_predictions[..., :2]
        objectness = raw_predictions[..., 2]
        shape_colors = raw_predictions[..., 3:6]
        letter_colors = raw_predictions[..., 6:9]
        shape_class_preds = raw_predictions[..., 9:9+self.num_classes]
        letter_class_preds = raw_predictions[..., 9+self.num_classes:]

        # boxes[..., :2] -= boxes[..., 2:] / 2 # adjust center coords to be top-left coords
        boxes*= 1/self.num_cells# scale to be percent of global image coords
        for i in range(boxes.shape[1]):# add offsets for each cell
            for j in range(boxes.shape[2]):
                boxes[..., i, j, 0] += j * (1/self.num_cells)
                boxes[..., i, j, 1] += i * (1/self.num_cells)
        # boxes[..., 2:] += boxes[..., :2] # add width and height to get bottom-right coords

        boxes = boxes.reshape(-1, 2)
        objectness = objectness.reshape(-1)
        shape_colors = shape_colors.reshape(-1, 3)
        letter_colors = letter_colors.reshape(-1, 3)
        shape_class_preds = shape_class_preds.reshape(-1, self.num_classes)
        letter_class_preds = letter_class_preds.reshape(-1, 36)

        return boxes, objectness, shape_colors, letter_colors, shape_class_preds, letter_class_preds 

    def predict(self, x: torch.Tensor, conf_threshold = 0.5, iou_threshold=0.5, max_preds = 10) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        '''
        Returns (boxes, objectness, shape_colors, letter_colors, shape_classes, letter_classes) where each is a tensor of shape (n, 4), (n, 1), (n,3), (n,3), (n, num_classes), (n, 36) respectively
        '''
        n_batches = x.shape[0]
        raw_predictions = self.forward(x)

        boxes, objectness, shape_colors, letter_colors, shape_classes, letter_classes = map(lambda x: x.to("cpu"), self.process_predictions(raw_predictions))
        # batch_indices = torch.arange(n_batches).repeat_interleave(boxes.shape[0]//n_batches).to(x.device)
        # batch_indices = batch_indices[objectness > conf_threshold]
        boxes = boxes[objectness > conf_threshold]
        shape_colors = shape_colors[objectness > conf_threshold]
        letter_colors = letter_colors[objectness > conf_threshold]
        shape_classes = shape_classes[objectness > conf_threshold]
        letter_classes = letter_classes[objectness > conf_threshold]

        # needs to be last or it fucks the indices for the rest of the predictions (getting punished for reassigning variables or something)
        objectness = objectness[objectness > conf_threshold]

        # kept_indices = batched_nms(boxes, objectness, batch_indices,iou_threshold) # todo: make this batched_nms and return the boxes per batch instead of as one

        # return boxes[kept_indices][:max_preds], classes[kept_indices][:max_preds], objectness[kept_indices][:max_preds]
        return boxes, objectness, shape_colors, letter_colors, shape_classes, letter_classes, raw_predictions
    
if __name__=="__main__":
    model = SUASYOLO(num_classes=14, img_width=640)
    input_shape = (1,3,640,640)
    summary(model, input_shape)