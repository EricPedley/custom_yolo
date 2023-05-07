from itertools import tee

import torch
import torch.nn as nn

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
            3, 64, 128, 256, 512, 1024
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


    def forward(self, x):
        x = self.feature_extraction(x)
        return self.detector(x)
