import torch
import torch.nn as nn
from torch.nn.modules.padding import ReflectionPad2d

"""
    In this module we define out models (Generator and Discriminator) building blocks.
"""

"""
    Implemetation of c7s1-k/dk in the article.
    Where c7s1-7 stands for 7x7 convolution-instancenorm-relu block with stride 1
    and k filters.
    dk stands for 3x3 convolution-instancenorm-relu with k filters and stride 2
"""


class ConvNormBlock(nn.Module):

    def __init__(self, in_channels, stride, kernel_size, k_filters=64, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels=k_filters, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(k_filters),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


"""
    Implemetation of Rk in the article.
    Where Rk stands for residual block with 2 conv-layers each with 3x3 filter 
"""


class ResBlock(nn.Module):
    def __init__(self, in_channels, k_filters):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels=k_filters, kernel_size=(3, 3)),
            nn.InstanceNorm2d(k_filters),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(k_filters, out_channels=in_channels, kernel_size=(3, 3)),
            nn.InstanceNorm2d(k_filters)


        )

    def forward(self, x):
        return x + self.block(x)


"""
    Implemetation of Uk in the article.
    Where Uk stands for fractionaly strided convolutional-instanceNorm-relu layer. 
"""


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, k_filters):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels=k_filters, kernel_size=(3, 3), stride=2),
            nn.InstanceNorm2d(k_filters),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)