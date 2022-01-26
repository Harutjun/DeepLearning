import os
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np



video = cv2.VideoCapture(r'video.mp4')



data_path = r'BWToColorData'
for i in range(5000):
    _, frame = video.read()
    
    
ret, frame = video.read()
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


idx = 1
max_l = int((video.get(cv2.CAP_PROP_FRAME_COUNT)- 5000)/ 30) 
#max_l = 10


for i in tqdm(range(max_l)):
    ret, frame = video.read()

    if not ret:
        continue
    
    for i in range(1):
        for j in range(2):
            cropped_f = frame[i*256: (i+1)*256, j*256: (j+1)*256]
            
            cv2.imwrite(os.path.join(data_path, f'color/{idx}.png'), cropped_f)
            
            gray_frame = cv2.cvtColor(cropped_f, cv2.COLOR_BGR2GRAY)
            
            cv2.imwrite(os.path.join(data_path, f'bw/{idx}.png'), gray_frame)
            # add noise
                        
            idx +=1
    
    # truncate frame
    for i in range(60):
        _, _ = video.read()
        
    max_l-=1
# plt.imshow(frame)
# plt.show()