import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Model.Generator import Generator
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


#device = 'cuda' if torch.cuda.is_availiable() else 'cpu'

state_dict_path = r'checkpoints/gen_g_color_to_gs.pt'

model = Generator(config=2, in_channels=3, out_im_ch=3)

model.load_state_dict(torch.load(state_dict_path))

model = model.eval()


#im = Image.open('BWToColorData/bw/1.png')
im = Image.open('BWToColorData/bw/31.png')
im_gs = ImageOps.grayscale(im)

trans_bw = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])
                            ])

inv_trans_color = transforms.Normalize(mean=[-1, -1, -1], 
                                       std=[1/0.5, 1/0.5, 1/0.5])

im_w, im_h = im.size
x_shapes = 5
y_shapes = 7

im_gs_tensor = trans_bw(im_gs)
new_im = torch.zeros((3, 7*256, 5*256))


im_tensor = torch.stack([im_gs_tensor, im_gs_tensor, im_gs_tensor], dim=1)

new_im = model(im_tensor).squeeze(0)
# for x in range(x_shapes):
#     for y in range(y_shapes):
        

#         im_tensor = im_gs_tensor[:, y*256: (y+1)*256, x*256: (x+1)*256]
#         im_tensor = torch.stack([im_tensor, im_tensor, im_tensor], dim=1)

#         out = model(im_tensor)
        
#         new_im[:, y*256: (y+1)*256, x*256: (x+1)*256] = out.squeeze(0)

            

# plt.imshow(im_gs)
# plt.show()

# trans_bw = transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Resize((256, 256)),
#                                 transforms.Normalize(mean=[0.5], std=[0.5])
#                             ])
# inv_trans_color = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])

# im_tensor = trans_bw(im_gs)
# im_tensor = torch.stack([im_tensor, im_tensor, im_tensor], dim=1)

# out = model(im_tensor)


out = inv_trans_color(new_im)

out_im = F.to_pil_image(out)
plt.imshow(out_im)
plt.show()