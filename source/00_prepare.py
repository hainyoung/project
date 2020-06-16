from PIL import Image
import glob
from natsort import natsorted
import os
import cv2
import numpy as np

##################################### 이미지 리사이즈
################ 1. 눈 감은 이미지 리사이즈
# display image characteristics
imc_path = './img/close/ce1.jpg' 
imc = Image.open(imc_path) 
print('{}'.format(imc.format)) 
print('size : {}'.format(imc.size))
print('image mode : {}'.format(imc.mode))
# imc.show()
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
# open_eyes image resize
imo_path = './img/open/oe1.jpg'
imo = Image.open(imo_path)
print('{}'.format(imo.format))
print('size : {}'.format(imo.size))
print('image mode : {}'.format(imo.mode))
# imo.show()
'''
JPEG
size : (299, 168)
image mode : RGB
'''

################ 1. 눈 뜬 이미지 리사이즈
# empty lists
open_list = []
resized_open = []
# append images to list
for filename in natsorted(glob.glob('./img/open/*.jpg')) :
    print(filename)
    imo = Image.open(filename)
    open_list.append(imo)
# append resized images to list
for imo in open_list :
    imo = imo.resize((64, 64))
    resized_open.append(imo)
    print('size : {}'.format(imo.size))
# save resized images to new folder
for (i, new) in enumerate(resized_open) :
    new.save ('{}{}{}'.format('./eyes/o_eyes/oe', i+1, '.jpg'))



############# 최종 예측에 사용할 x_pred(눈 뜬 사진들로만 준비)
imt_path = './img/test/test1.jpg' 
imt = Image.open(imt_path) 
print('{}'.format(imt.format)) 
print('size : {}'.format(imt.size))
print('image mode : {}'.format(imt.mode))
# imc.show()
test_list = []    
resized_test = []  
# append images to list
for filename in natsorted(glob.glob('./img/test/*.jpg')) :
    print(filename) 
    imt = Image.open(filename) 
    test_list.append(imt)    
# append resized images to list
for imt in test_list :       
    imt = imt.resize((64, 64))
    resized_test.append(imt)  
    print('size : {}'.format(imt.size))
# save resized images to new folder
for (i, new) in enumerate(resized_test) :
    new.save ('{}{}{}'.format('./data/test_face/tf', i+1, '.jpg')) 
      

######################## NUMPY 변환
# 준비한 이미지 numpy 형태 dataset 만들기
groups_folder_path = './eyes/'     
categories = ["c_eyes", "o_eyes"]  
num_classes = len(categories)
print(num_classes) # 2
# x, y data준비
x = [] # 빈 리스트
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


# x_pred data 준비
groups_folder_path = './data/test_face/'     
# x, y data준비
x_pred = [] # 빈 리스트
image_dir = groups_folder_path +  '/'
for path, dirs, files in os.walk(image_dir) :
    for filename in files :
        print(image_dir + filename)
        img = cv2.imread(image_dir+filename)
        x_pred.append(img)
        # print(x_pred)

x_pred = np.array(x_pred)

# 64 x 64
print("x.shape :", x.shape)   # (200, 64, 64, 3)
print("y.shape :", y.shape)   # (200, 2)

# 100 x 100 
# print("x.shape :", x.shape)   # (200, 100, 100, 3)
# print("y.shape :", y.shape)   # (200, 2)

print("x_pred.shape :", x_pred.shape)


# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)
np.save('./data/x_pred.npy', x_pred)
