from PIL import Image
import glob
from natsort import natsorted
import numpy as np
import os
import cv2

# 1. image resize
# closed eyes resize
close_list = []    
resized_close = []  
# append images to list
for filename in natsorted(glob.glob('./img/close/*.jpg')) :
    print(filename) 
    imc = Image.open(filename) 
    close_list.append(imc)    
# append resized images to list
for imc in close_list :       
    imc = imc.resize((64, 64))
    resized_close.append(imc)  
    print('size : {}'.format(imc.size))
# save resized images to new folder
for (i, new) in enumerate(resized_close) :
    new.save ('{}{}{}'.format('./eyes/close/ce', i+1, '.jpg')) 













# open eyes resize
open_list = []
resized_open = []

for filename in natsorted(glob.glob('./img/open/*.jpg')) :
    # print(filename)
    imo = Image.open(filename)
    open_list.append(imo)

for imo in open_list :
    imo = imo.resize((64, 64))
    resized_open.append(imo)
    # print('size : {}'.format(imo.size))

for (i, new) in enumerate(resized_open) :
    new.save ('{}{}{}'.format('./eyes/open/oe', i+1, '.jpg'))











# 2. 이미지 dataset 만들기
groups_folder_path = './eyes/'     
categories = ["close", "open"]  
num_classes = len(categories)
print(num_classes) # 2

x = []
y = []

for index, categorie in enumerate(categories) :
    label = [0 for i in range(num_classes)]
    label[index] = 1
    image_dir = groups_folder_path + categorie + '/'

    for path, dirs, files in os.walk(image_dir) :
        for filename in files :
            print(image_dir + filename)
            img = cv2.imread(image_dir+filename)
            x.append(img)
            y.append(label)

x = np.array(x)
y = np.array(y)

print("x.shape :", x.shape)   # (200, 64, 64, 3)
print("y.shape :", y.shape)   # (200, 2)

# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)


