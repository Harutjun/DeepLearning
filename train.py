import itertools

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


from Generator import Generator
from Discriminator import Discriminator
from ShoeEdgeDataset import ShoeEdgeDataset

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image
import numpy as np


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/CycleGan')

BATCH_SIZE = 1
lr = 2e-4
PRINT_FREQ = 20
n_epochs = 50
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


train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((128, 128)),
                                        ])

dsEdgeToShoe = ShoeEdgeDataset(r'Data/train',r'Data/val', False, transform=train_transforms)
#dsShoeToEdge = ShoeEdgeDataset(r'Data/train',r'Data/val', True)





edgeToShoeLoader = DataLoader(dsEdgeToShoe, batch_size=BATCH_SIZE, shuffle=True)
#ShoeToEdgeLoader = DataLoader(dsShoeToEdge, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

def _init_weights(m):
    def init_func(m):  # define the initialization function
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



# transformation from A to B (G: X->Y)
genG = Generator(1, 3).to(DEVICE)

# transformation for B to A (F: Y->X)
genF = Generator(1, 3).to(DEVICE)

# discriminate between fake Y and Y
discY = Discriminator().to(DEVICE)

# discriminate between fake X and X
discX = Discriminator().to(DEVICE)


genG.apply(_init_weights)
genF.apply(_init_weights)
discY.apply(_init_weights)
discX.apply(_init_weights)



optimGen = optim.Adam(itertools.chain(genG.parameters(), genF.parameters()), lr = lr, betas=(0.5, 0.999))
optimDiscY = optim.Adam(discY.parameters(), lr = lr, betas=(0.5, 0.999))
optimDiscX = optim.Adam(discX.parameters(), lr = lr, betas=(0.5, 0.999))


criterionGan = torch.nn.MSELoss()
criterionCycle = torch.nn.L1Loss()

for epoch in tqdm(range(n_epochs)):
    
    for batchIDX, (domainA, domainB) in enumerate(edgeToShoeLoader):
        
        # domain A is edges (X)
        # domain B is shoes (Y)
        
        x, y = domainA.to(DEVICE), domainB.to(DEVICE)
        
        
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
        
        
        
        if (batchIDX+1) % PRINT_FREQ == 0:
            writer.add_scalar('Generator loss', lossG.item(), batchIDX+1)
            writer.add_scalar('Discriminator X loss', discXloss.item(), batchIDX+1)
            writer.add_scalar('Discriminator Y loss', discYloss.item(), batchIDX+1)
            
            writer.add_image('Source image', x[0, ...].to('cpu'), (batchIDX+1))
            writer.add_image('Target image', y[0, ...].to('cpu'), (batchIDX+1))
            writer.add_image('Generated image', fake_y[0, ...].to('cpu'), (batchIDX+1))
