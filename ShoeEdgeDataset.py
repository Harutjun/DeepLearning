import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image


""" 
    Implementation of shoes edges to shoes dataset
"""
class ShoeEdgeDataset(Dataset):
    
    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
        self.val_dir = val_dir
        
        
        # load all the file names
        self.file_names = os.listdir(train_dir)
        
    
    def __getitem__(self, index):
        im = Image.open(os.path.join(self.train_dir, self.file_names[index]))
        
        im_width, im_height = im.size
        
        # take the first half of the image as source
        src_im = im.crop((0,0, im_width // 2, 0)) 
        
        # set the other half as target
        target_im = im.crop((im_width // 2, 0, 0, 0))
        
        return src_im, target_im
    
    
    def len(self):
        return len(self.file_names)
        
        