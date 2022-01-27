import itertools
from time import sleep
from matplotlib.colors import Normalize

import torch
import torch.nn as nn

import torchvision
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms.functional as FuncTrans

from Model.Generator import Generator
from Model.Discriminator import Discriminator
from Datasets.BWToColorData import BWToColorDS

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image
import numpy as np
from Utils import get_ds_stats

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/BWToColor_gs')

BATCH_SIZE = 1
lr = 2e-4
PRINT_FREQ = 20
n_epochs = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
disc_update_freq = 5
lamda = 10

print(f"running on: {DEVICE}")

#################################
# Set up data set & loaders     #
#################################


#################################
# Set up image transformation   #
#################################

# The normalization parameters are calculated before hand using our BwToColorDS._get_ds_stats() method.

# trans_color = transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Resize((256, 256)),
#                                 transforms.Normalize(mean=[0.3223, 0.3353, 0.3270], std=[0.2429, 0.2512, 0.2716])
#                                 ])

trans_color = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])


# trans_bw = transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Resize((256, 256)),
#                             transforms.Normalize(mean=[0.3304], std=[0.4703])
#                             ])

trans_bw = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((256, 256)),
                            transforms.Normalize(mean=[0.5], std=[0.5])
                            ])


# trans_cityscape = transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Resize((256, 256)),
#                             ])


inv_trans_bw = transforms.Normalize(mean=[-1], std=[1/0.5])

inv_trans_color = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])
                                      
dsColorToGray = BWToColorDS(r'BWToColorData/bw', r'BWToColorData/color', trans_color=trans_color, trans_bw=trans_bw)



gsToColor_loader = DataLoader(dsColorToGray, batch_size=BATCH_SIZE, shuffle=True)


def _init_weights(m):
    def init_func(m):  # define the initialization function
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



# transformation from A to B (G: X->Y)
genG = Generator(config = 2, in_channels = 3, out_im_ch=3).to(DEVICE)

# transformation for B to A (F: Y->X)
genF = Generator(config = 2, in_channels = 3, out_im_ch=3).to(DEVICE)

# discriminate between fake Y and Y
discY = Discriminator(in_channels=3).to(DEVICE)

# discriminate between fake X and X
discX = Discriminator(in_channels=3).to(DEVICE)

genG.apply(_init_weights)
genF.apply(_init_weights)
discY.apply(_init_weights)
discX.apply(_init_weights)



optimGen = optim.Adam(itertools.chain(genG.parameters(), genF.parameters()), lr = lr, betas=(0.5, 0.999))
optimDiscY = optim.Adam(discY.parameters(), lr = lr, betas=(0.5, 0.999))
optimDiscX = optim.Adam(discX.parameters(), lr = lr, betas=(0.5, 0.999))



criterionGan = torch.nn.MSELoss()
criterionCycle = torch.nn.L1Loss()

best_discX_loss = 1e9
best_discY_loss = 1e9
for epoch in tqdm(range(n_epochs)):
    running_discY_loss = 0
    running_discX_loss = 0
    for batchIDX, (domainA, domainB) in enumerate(gsToColor_loader):
        
        # domain A is BW (X)
        # domain B is Color (Y)
        
        x, y = domainA.to(DEVICE).squeeze_(1), domainB.to(DEVICE)
                    
        optimGen.zero_grad()
        # Gan loss
        fake_y = genG(x)
        discY_out = discY(fake_y)
        
        lossGanA = criterionGan(discY_out, torch.ones_like(discY_out))
        
        fake_x = genF(y)
        discX_out = discX(fake_x)
        
        lossGanB = criterionGan(discX_out, torch.ones_like(discY_out))
        
        # end gan loss
        
        # Cycle loss
        
        recoverd_x = genF(fake_y)
        
        recoverd_y = genG(fake_x)
        
        cycleLoss = lamda*(criterionCycle(recoverd_x, x) + criterionCycle(recoverd_y, y))
        
        lossG = lossGanA + lossGanB + cycleLoss
        lossG.backward()
        
        
        optimGen.step()
                
        
        ###################
        #  Train disc Y   #
        ###################
        optimDiscY.zero_grad()

        fakeY = genG(x)
        
        # fake update
        discYfake = discY(fakeY.detach())
        
        # real update
        discYreal = discY(y)
        
        
        discYloss = 0.5 * (criterionGan(discYreal, torch.ones_like(discYreal)) + \
                           criterionGan(discYfake, torch.zeros_like(discYfake)))
        
        discYloss.backward()
        optimDiscY.step()
        
        
        ###################
        #   Train disc X  #
        ###################
        optimDiscX.zero_grad()

        fakeX = genF(y)
        
        # fake update
        discXfake = discX(fakeX.detach())
        
        # real update
        discXreal = discX(x)
        
        
        discXloss = 0.5 * (criterionGan(discXreal, torch.ones_like(discXreal)) + \
                          criterionGan(discXfake, torch.zeros_like(discXfake)))
        
        discXloss.backward()
        optimDiscX.step()
        
        running_discY_loss += discYloss.item()
        running_discX_loss += discXloss.item()
        
        if (batchIDX+1) % PRINT_FREQ == 0:
            
            avg_discX = running_discX_loss / PRINT_FREQ
            avg_discY = running_discY_loss / PRINT_FREQ
            
            if avg_discY < best_discY_loss:
                best_discY_loss = avg_discY
                torch.save(genG.state_dict(), r'checkpoints/gen_g_color_to_gs.pt')
                sleep(3)
            
                
            
            writer.add_scalar('Generator loss', lossG.item(), batchIDX+1 + epoch*len(gsToColor_loader))
            writer.add_scalar('Discriminator Y loss', discYloss.item(), batchIDX+1 + epoch*len(gsToColor_loader))
            
            
            writer.add_image('Source image', inv_trans_color(x[0, ...]).to('cpu'), (batchIDX+1 + epoch*len(gsToColor_loader)))
            writer.add_image('Target image', inv_trans_color(y[0, ...]).to('cpu'), (batchIDX+1 + epoch*len(gsToColor_loader)))
            writer.add_image('Generated image', inv_trans_color(fake_y[0, ...]).to('cpu'), (batchIDX+1 + epoch*len(gsToColor_loader)))

    torch.save(genG.state_dict(), r'checkpoints/gen_g_color_to_gs_latest.pt')
    sleep(3)