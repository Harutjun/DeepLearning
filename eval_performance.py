import torch
import torch.nn as nn
from Datasets.ShoeEdgeDataset import ShoeEdgeDataset
from Model.Generator import Generator
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from Datasets.BWToColorData import BWToColorDS


def plot_sample(im_tensor, gt_tensor, sktech_tensor):
    import torchvision.transforms.functional as F
    de_norm = transforms.Normalize([-1, -1, -1], [2, 2, 2])
    
    
    im = F.to_pil_image(de_norm(im_tensor.squeeze(0)))
    gt_im = F.to_pil_image(de_norm(gt_tensor.squeeze(0)))
    sktech_im = F.to_pil_image(de_norm(sktech_tensor.squeeze(0)))
    
    
    plt.subplot(1,3,1)
    plt.imshow(sktech_im)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(gt_im)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(im)
    plt.axis('off')


    plt.show()
    
    
    
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _init_gen(state):
        gen = Generator(config=2, in_channels=3, out_im_ch=3)
        
        gen.load_state_dict(torch.load(state))
        gen.eval()
        
        return gen.to(device)


gen = _init_gen(r'checkpoints/gen_g_color_to_gs_latest.pt')
transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((256, 256)),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trans_bw = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((256, 256)),
                                        transforms.Normalize((0.5), (0.5))])

dsColorToGray = BWToColorDS(r'BWToColorData/bw', r'BWToColorData/color', trans_color=transform, trans_bw=trans_bw, max_items=200)


loader = DataLoader(dsColorToGray, batch_size=1)


image_dim = 256 * 256 * 3

avg_pixel_acc = 0
for batchIDX, (domainA, domainB) in tqdm(enumerate(loader)):
    domainA, domainB = domainA.to(device).squeeze(0), domainB.to(device)
    
    im_tensor = torch.stack([domainA, domainA, domainA], dim=1)
    gen_image = gen(domainA)
    #plot_sample(gen_image, domainB, domainA)
    gen_image = (255 * gen_image).int()
    
    
    domainB = (255 * domainB).int()
    # calculate pixel accuracy
    correct_pixel = (torch.abs(gen_image- domainB) < 10).count_nonzero()
    
    avg_pixel_acc += (correct_pixel / image_dim)


avg_pixel_acc /= len(dsColorToGray)


print(f"per pixel accuracy: {avg_pixel_acc:.4f}")