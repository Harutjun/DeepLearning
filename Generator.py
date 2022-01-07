import torch
import torch.nn as nn
from torch.nn.modules.padding import ReflectionPad2d
import BuildingBlock


"""
    In this module we define the Generator class
"""


"""
    The generator had 2 possible structures as described in the article:
    1) 6 Res blocks : c7s1-64,d128,d256,  6 X R256 , u128,u64,c7s1-3
    2) 9 Res blocks : c7s1-64,d128,d256,  9 X R256 , u128,u64,c7s1-3  
    
    (1) is used for 128x128 training images
    (2) is used for 256x256 or higher training images
 
   where c7s1-k is 7x7-k Conv with stride = 1 , Instance Norm, ReLU Layer  and k filters          
   dk is a 3x3 Conv_InstanceNorm-ReLU layer with k Channels and Stride = 2 (Downsampling Layer)
   Rk is a Res Block with two 3x3 Conv Layers with k filters on each layer
   uk is a 3x3 fraction-strided Conv - instanceNorm - ReLU layer with k filters and stride = 1/2 (Upsampling)
    
"""

class Generator(nn.Module):

    def __init__(self, config, in_channels):
        """
        config = 1 :  6 resnet blocks
        config = 2 :  9 resnet blocks

        """
        ResLayers = 3 + 3 * config
        super().__init__()
        self.layers = self.__init__layers(ResLayers, in_channels)
        
        
    def _init_weights():
        def init_func(m):  # define the initialization function
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
            
    def __init__layers(selfself, ResLayers, in_channels):
        """
            Internal method to generate layers from layer's config list.
        Args:
            config (list): list of string representing layers
        """

        layers = nn.ModuleList()
        layers += [BuildingBlock.ConvNormBlock(in_channels, stride=1, kernel_size=(7, 7), k_filters=64, padding=3)]
        layers += [BuildingBlock.ConvNormBlock(64, stride=2, kernel_size=(3, 3),  k_filters=128)]
        layers += [BuildingBlock.ConvNormBlock(128, stride=2, kernel_size=(3, 3),  k_filters=256)]

        for layer in range(ResLayers):
            layers += [BuildingBlock.ResBlock(in_channels=256, k_filters=256)]

        layers += [BuildingBlock.UpsampleBlock(256, k_filters=128)]
        layers += [BuildingBlock.UpsampleBlock(128, k_filters=64)]
        layers += [BuildingBlock.ConvNormBlock(64, stride=1, kernel_size=(7, 7),  k_filters=3, padding=3, last=True)]

        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def test_generator():

    config = 2
    gen = Generator(config=config, in_channels=3)


    sample = torch.rand((2, 3, 256, 256))
    out = gen(sample)

    print(out.shape)
    #print(gen.layers)

#test_generator()

