import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps


""" 
    Implementation of shoes edges to shoes dataset
"""
class ShoeEdgeDataset(Dataset):
    
    def __init__(self, train_dir, val_dir, flip, transform=None, max_items = 0):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.flip = flip
        self.transform = transform
        
        if max_items == 0:
            max_items = len(os.listdir(train_dir))
            
        # load all the file names
        self.file_names = os.listdir(train_dir)[:int(max_items)]
        
    
    def __getitem__(self, index):
        im = Image.open(os.path.join(self.train_dir, self.file_names[index]))
        
        im_width, im_height = im.size
        
        # take the first half of the image as source
        
        src_im = im.crop((0,0, im_width // 2, im_height))
        #src_im = ImageOps.grayscale(src_im)
        
        
        # set the other half as target
        target_im = im.crop((im_width // 2, 0, im_width, im_height))
        
        if self.flip:
            src_im, target_im = target_im, src_im
        
        if self.transform:
            return self.transform(src_im), self.transform(target_im)
        
        return src_im, target_im
    
    
    def __len__(self):
        return len(self.file_names)
        
def test_dataset():
    ds = ShoeEdgeDataset(r'Data/train', r'Data/val', flip=False)
    
    import matplotlib.pyplot as plt
    
    im, targ = ds.__getitem__(10)
    
    plt.subplot(1,2,1)
    plt.imshow(im, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(targ)
    plt.show()

#test_dataset()