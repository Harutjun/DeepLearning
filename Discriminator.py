import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Building block of out discriminator
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,use_instance_norm=True):
        super().__init__()
        self.norm = use_instance_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1, padding_mode='reflect')
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        if self.norm:
            self.instance_norm = nn.InstanceNorm2d(out_channels)
    
    def forward(self, x):
        if self.norm:
            return self.leaky_relu(self.instance_norm(self.conv(x)))
        else:
            return self.leaky_relu(self.conv(x))


"""
    Implemtation of the discrimintor from the configuration given by out article.
"""
class Discriminator(nn.Module):
   
    def __init__(self, config, in_channels):
        super().__init__()

        self.layers = self._init_layers(config, in_channels)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x

    

    def _init_layers(self, config: list, in_channels):
        """
            Internal method to generate layers from layer's config list.

        Args:
            config (list): list of string representing layers
        """
        layers = nn.ModuleList()
        
        curr_in_channels = in_channels
        
        for layer in config:
            of_filters = int(layer[1:])
            
            if of_filters == 64:
                layers += [ConvBlock(curr_in_channels, of_filters, use_instance_norm=False)]
            
            else:
                layers += [ConvBlock(curr_in_channels, of_filters)]
            
            curr_in_channels = of_filters
        
        
        last_conv = nn.Conv2d(curr_in_channels, 1, (4, 4), stride= 1, padding=1, padding_mode='reflect')
        
        layers += [last_conv]
        return layers
            




def test_discriminiator():
    cfg = [
        "C64",
        "C128",
        "C256",
        "C512"     
    ]
    disc = Discriminator(cfg, 3)
    
    sample = torch.rand((2, 3, 256, 256))
    
    out = disc(sample)
    
    print(out.shape)
    
test_discriminiator()