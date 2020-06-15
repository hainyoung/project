import os, re, glob
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# 준비한 이미지 numpy dataset 만들기
groups_folder_path = './eyes/'     # dataset으로 사용할 파일들의 가장 상위폴더 지정
categories = ["c_eyes", "o_eyes"]  # y data의 카테고리 지정

num_classes = len(categories)
print(num_classes) # 2

# x, y data준비
x = [] # 빈 리스트
y = []

for index, categorie in enumerate(categories) :
    label = [0 for i in range(num_classes)]
    label[index] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir) :
        for filename in f :
            print(image_dir + filename)
            img = cv2.imread(image_dir+filename)
            x.append(img)
            y.append(label)

x = np.array(x)
y = np.array(y)

# 64 x 64
print("x.shape :", x.shape)   # (200, 64, 64, 3)
print("y.shape :", y.shape)   # (200, 2)
 
# print("x.shape :", x.shape)   # (200, 100, 100, 3)
# print("y.shape :", y.shape)   # (200, 2)
 
# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)

