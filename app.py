from dis import show_code
import tkinter
from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter import filedialog as fd

from PIL import ImageGrab, ImageOps, ImageTk, Image
import matplotlib.pyplot as plt

import torch 
import torch.nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from Model.Generator import Generator

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def _init_gen(self, state):
        gen = Generator(1, in_channels=3, out_im_ch=3)
        
        gen.load_state_dict(torch.load(state))
        gen.eval()
        
        return gen
    
    def _prepare_image(self, im: Image):
        transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((128, 128)),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        im_tensor = transform(im)
        im_tensor.unsqueeze_(0)
        return im_tensor
        
        
    def __init__(self):
        self.gen = self._init_gen(r'checkpoints/gen_g_latest_gs.pt')
        self.gen2 = self._init_gen(r'checkpoints/gen_f_latest_gs.pt')
        
        self.inv_trans = transforms.Normalize(
                                mean=[-1, -1, -1],
                                std=[1/0.5, 1/0.5, 1/0.5]
                            )
        
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)
        
        
        self.clear_button = Button(self.root, text='clear', command= self.use_clear)
        self.clear_button.grid(row=0, column=6)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)
        
        
        self.eraser_button = Button(self.root, text='Load image', command=self.load_image)
        self.eraser_button.grid(row=0, column=7)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=256, height=256)
        self.gen_c = Canvas(self.root, bg='white', width=256, height=256)
        self.gen_edges = Canvas(self.root, bg='white', width=256, height=256)
        
        self.c.grid(row=1, columnspan=5)
        self.gen_c.grid(row=1, column = 5, columnspan=5)
        self.gen_edges.grid(row=1, column = 10, columnspan=5)
        
        self.generate_button = Button(self.root, text='Generate', command=self.use_generate)
        self.generate_button.grid(row=0, column=5)
        
        self.generate_button = Button(self.root, text='To Edges !', command=self.cvt_to_edges)
        self.generate_button.grid(row=0, column=8)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
    
    def cvt_to_edges(self):
        im = self.show_canvas(self.gen_c)
        
        im_net = self._prepare_image(im)
        
        
        out_im = self.gen2(im_net)
        
        out_im = self.inv_trans(out_im)
        out_im = out_im.squeeze(0)
        shoe_im = F.to_pil_image(out_im)
        
        shoe_im = shoe_im.resize((248,248))
        photo = ImageTk.PhotoImage(image =shoe_im)
        self.gen_edges.create_image(0, 0, image=photo, anchor=tkinter.NW)
        #self.label.place(x=240, y=50)
        
        self.root.mainloop()

        
    def use_generate(self):
        im = self.show_canvas(self.c)
        
        im_net = self._prepare_image(im)
        
        
        out_im = self.gen(im_net)
        
        out_im = self.inv_trans(out_im)
        out_im = out_im.squeeze(0)
        shoe_im = F.to_pil_image(out_im)
        
        
        shoe_im = shoe_im.resize((248,248))
        photo = ImageTk.PhotoImage(image =shoe_im)
        self.gen_c.create_image(0, 0, image=photo, anchor=tkinter.NW)
        
        self.root.mainloop()


    def use_clear(self):
        self.c.delete("all")
        self.gen_c.delete("all")
        self.gen_edges.delete("all")
        
    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    
    def load_image(self):
        filename = fd.askopenfilename()

        im = Image.open(filename)
        im_width, im_height = im.size
        src_im = im.crop((0,0, im_width // 2, im_height))
        
        
        src_im = src_im.resize((248,248))
        photo = ImageTk.PhotoImage(image =src_im)
        self.c.create_image(0, 0, image=photo, anchor=tkinter.NW)
        self.root.mainloop()
    
    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    
    
    def show_canvas(self, c):
        x=self.root.winfo_rootx()+self.c.winfo_x()
        y=self.root.winfo_rooty()+self.c.winfo_y()
        x1=x+self.c.winfo_width()
        y1=y+self.c.winfo_height()
        
        
        im = ImageGrab.grab().crop((x,y,x1,y1))
        # plt.imshow(im)
        # plt.show()
        return im
    
if __name__ == '__main__':
    Paint()