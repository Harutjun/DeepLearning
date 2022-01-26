from email.mime import image
import torch
import torch.nn as nn
from Datasets.ShoeEdgeDataset import ShoeEdgeDataset
from Model.Generator import Generator
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _init_gen(state):
        gen = Generator(1, in_channels=3, out_im_ch=3)
        
        gen.load_state_dict(torch.load(state))
        gen.eval()
        
        return gen.to(device)


gen = _init_gen(r'checkpoints/gen_g_latest_gs.pt')
transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((128, 128)),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ShoeEdgeDataset(train_dir='Data/train', val_dir = r'Data/val', flip=False, transform = transform, val=True)


edgeToShoeLoader = DataLoader(dataset, batch_size=1)


image_dim = 128 * 128 * 3

avg_pixel_acc = 0
for batchIDX, (domainA, domainB) in tqdm(enumerate(edgeToShoeLoader)):
    domainA, domainB = domainA.to(device), domainB.to(device)
    gen_image = gen(domainA)
    
    gen_image = (255 * gen_image).int()
    domainB = (255 * domainB).int()
    # calculate pixel accuracy
    correct_pixel = (torch.abs(gen_image- domainB) < 0).count_nonzero()
    
    avg_pixel_acc += (correct_pixel / image_dim)


avg_pixel_acc /= len(dataset)


print(f"Pixel accuracy if: {avg_pixel_acc:.4f}")