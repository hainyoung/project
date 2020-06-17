from PIL import Image
import glob
from natsort import natsorted
import os
import cv2
import numpy as np

##################################### 이미지 리사이즈
################ 1. closed eyes resize
# display image characteristics
imc_path = './img/close/ce1.jpg' 
imc = Image.open(imc_path) 
print('{}'.format(imc.format)) 
print('size : {}'.format(imc.size))
print('image mode : {}'.format(imc.mode))
imc.show()
'''
JPEG
size : (852, 480)
image mode : RGB
'''
# close_eyes image resize
# empty lists
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
    new.save ('{}{}{}'.format('./eyes/c_eyes/ce', i+1, '.jpg')) 

################ 2. open eyes resize
# empty lists
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
    new.save ('{}{}{}'.format('./eyes/o_eyes/oe', i+1, '.jpg'))



############# x_pred(open eyes)
imt_path = './img/test/test1.jpg' 
imt = Image.open(imt_path) 

test_list = []    
resized_test = []  

for filename in natsorted(glob.glob('./img/test/*.jpg')) :
    print(filename) 
    imt = Image.open(filename) 
    test_list.append(imt)    

for imt in test_list :       
    imt = imt.resize((64, 64))
    resized_test.append(imt)  
    # print('size : {}'.format(imt.size))

for (i, new) in enumerate(resized_test) :
    new.save ('{}{}{}'.format('./data/test_face/tf', i+1, '.jpg')) 
      

######################## NUMPY conversion
# numpy type dataset
groups_folder_path = './eyes/'     
categories = ["c_eyes", "o_eyes"]  
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

# 64 x 64
print("x.shape :", x.shape)   # (200, 64, 64, 3)
print("y.shape :", y.shape)   # (200, 2)

# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)
