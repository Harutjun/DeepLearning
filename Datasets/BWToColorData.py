import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import cv2

class BWToColorDS(Dataset):
    
    def __init__(self, train_bw, train_color, trans_color=None, trans_bw=None,max_items = 0):
        self.train_color = train_color
        self.train_bw = train_bw
        self.trans_color = trans_color
        self.trans_bw = trans_bw
        
        if max_items == 0:
                        
            self.len = len(os.listdir(train_bw))
            
            self.black_fn = os.listdir(train_bw)
            self.color_fn = os.listdir(train_color)
    
    
    def __getitem__(self, index):
        im_gray = Image.open(os.path.join(self.train_bw, self.black_fn[index]))
        im_color = Image.open(os.path.join(self.train_color, self.color_fn[index]))
        
        
        if self.trans_bw:
            gray_trans = self.trans_bw(im_gray)
            
            
            gray_trans = torch.stack([gray_trans, gray_trans, gray_trans], dim=1)
            return gray_trans, self.trans_color(im_color)
        
        return im_gray, im_color
    
    
    def __len__(self):
        return self.len
    
    def _get_ds_stats(self):
        # calculate mean and std for the dataset
        
        import torchvision.transforms
        
        to_tensor = torchvision.transforms.ToTensor()
        
        
        n_samples = self.__len__()
        color_ch_sum = 0
        color_ch_sqrd_sum = 0
        
        bw_ch_sum = 0
        bw_ch_sqrd_sum = 0
        
        from tqdm import tqdm
        for ii in tqdm(range(n_samples)):
            bw, color = self.__getitem__(ii)
            
            bw = to_tensor(bw)
            color = to_tensor(color)
            
            # mean across W,H
            color_ch_sum += torch.mean(color, dim=[1,2])
            color_ch_sqrd_sum += torch.mean(color**2, dim=[1,2])
            
            bw_ch_sum += torch.mean(bw)
            bw_ch_sqrd_sum += torch.mean(bw)
            
            
            
        mean_c = color_ch_sum / n_samples
        mean_bw = bw_ch_sum / n_samples

        # std = sqrt(E[X^2] - (E[X])^2)
        std_C = (color_ch_sqrd_sum / n_samples - mean_c ** 2) ** 0.5
        std_bw = (bw_ch_sqrd_sum / n_samples - mean_bw ** 2) ** 0.5
        
        
        return (mean_c, std_C) , (mean_bw, std_bw)

            
    


def test_ds():
    ds = BWToColorDS(r'BWToColorData/bw', r'BWToColorData/color')
    
    import matplotlib.pyplot as plt
    
    gray, color = ds.__getitem__(10)
    
    norm_c, norm_gs = ds._get_ds_stats()
    
    print(norm_c)
    print(norm_gs)
    
    plt.subplot(1,2,1)
    plt.imshow(gray, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(color)
    plt.show()


#test_ds()