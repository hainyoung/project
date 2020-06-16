# # resize

from PIL import Image

# pip install Pillow로 
# 파이썬에서 이미지를 처리하고 다루기 위한 외부 패키지 설치
# Pillow는 파이썬 이미징 라이브러리로서 여러 이미지 파일 포맷을 지원하고
# 이미지 내부 데이터에 접근할 수 있도록 한다
# ex) 이미지로부터 Thumbnail 이미지를 만든다거나
# 다른 이미지 포맷으로 변환할 수도 있고 이미지를 출력하는 일들을 할 수 있다
# 또한 이미지 크기를 변형하거나 회전 및 변형, 필터링 등 다양한 이미지 프로세싱 작업들을 할 수 있다

import glob
# 폴더 내 정렬 순서 그대로 정렬된다

from natsort import natsorted
# natsorted : to maintatin the order of my filenames
# 만들어놓은 폴더 내 정해둔 파일 순서로 정렬된다

# display image characteristics
# 사용할 이미지 특성 파악
imc_path = './img/close/ce1.jpg'  # 특성을 파악할 이미지 경로를 변수에 대입

imc = Image.open(imc_path) # 특정한 이미지 경로를 통해 해당 이미지를 open, 열어서 imc 변수에 대입

# format()을 사용한 문자열 formatting(포맷팅)을 사용

# print('{}'.format(imc.format)) # 지정해 둔 변수, 이미지의 형식 출력
# print('size : {}'.format(imc.size)) # 이미지 사이즈 출력
# print('image mode : {}'.format(imc.mode)) # 이미지의 색상 모드 출력 

# imc.show()
'''
JPEG 
size : (852, 480)
image mode : RGB
'''

# close_eyes image resize

# empty lists
close_list = []     # 사용할 이미지들을 담아 둘 빈 list 생성
resized_close = []  # resize 한 후 이미지들을 담아 둘 빈 list 생성 

# append images to list
for filename in natsorted(glob.glob('./img/close/*.jpg')) : # glob.glob : 특정 폴더 내의 특정 형식의 파일들을 불러온다
    print(filename) # 파일명뿐만 아니라 파일경로까지 함께 출력된다(glob.glob 특징)
    imc = Image.open(filename) # Image.open : 기존 이미지 파일 열 때 사용
    close_list.append(imc)     # 기존 이미지 파일 imc를 close_list라는 변수(빈 리스트)에 append

# append resized images to list
for imc in close_list :        # close_list에 있는 이미지들 for문으로 resize 실행
    imc = imc.resize((64, 64)) # 64 x 64로 resize
    resized_close.append(imc)  # resize_close라는 빈 리스트에 resize 된 이미지들(imc)를 append
    print('size : {}'.format(imc.size))

# save resized images to new folder
for (i, new) in enumerate(resized_close) :
    new.save ('{}{}{}'.format('./eyes/c_eyes/ce', i+1, '.jpg')) 
# resize 된 이미지들의 리스트를 새로운 장소에 새로운 이름으로 저장(save)
# 파일경로 / 파일넘버(1부터 시작하도록 i+1) / 파일확장자

# enumerate 함수
# 리스트가 있는 경우, 순서와 리스트의 값을 전달하는 기능을 가짐
# enumearte : 열거하다
# 이 함수는 순서가 있는 자료형(list, set, tuple, dictionary, string)을
# 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다
# ex)
# for i, name in enumerate(['body', 'foo', 'bar']) :
#     print(i, name)
'''
0 body
1 foo
2 bar
'''


# # open_eyes image resize

imo_path = './img/open/oe1.jpg'

imo = Image.open(imo_path)
# print('{}'.format(imo.format))
# print('size : {}'.format(imo.size))
# print('image mode : {}'.format(imo.mode))
# imo.show()

# '''
# JPEG
# size : (299, 168)
# image mode : RGB
# '''

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



# npy
import os, re, glob
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# 준비한 이미지 numpy dataset 만들기(카테고리, 인덱스 라벨링)
groups_folder_path = './eyes/'     # dataset으로 사용할 파일들의 가장 상위폴더 지정
categories = ["c_eyes", "o_eyes"]  # 

num_classes = len(categories)
# print(num_classes) # 2

# x, y data준비
x = [] # 빈 리스트
y = []

# 라벨링 코드 분석
for index, categorie in enumerate(categories) :
    # print(index) 
    
    # 0
    # 1
    
    label = [0 for i in range(num_classes)]
    # print(label)
    # [0, 0]
    # [0, 0]
    
    label[index] = 1
    # print(label)
    # [1, 0]
    # [0, 1]
    
    image_dir = groups_folder_path + categorie + '/'
    # print(image_dir)
    # ./eyes/c_eyes/
    # ./eyes/o_eyes/


    for path, dirs, files in os.walk(image_dir) : 
        # path : dir과 files가 있는 경로, dirs : path 아래에 있는 폴더, files : path 아래에 있는 파일들
        # os.walk : 시작 디렉토리부터 하위 모든 디렉토리를 차례대로 방문 해주는 함수
        for filename in files :
            print(image_dir + filename) # 파일들의 경로 확인
            img = cv2.imread(image_dir+filename) # img 라는 변수에 cv2를 이용해 이미지 파일들을 읽음
            x.append(img)   # x에 읽은 이미지들을 append
            y.append(label) # y에 label append


# 이미지 파일을 numpy 로 변환
x = np.array(x)
y = np.array(y)

# resize 64 x 64
# print("x.shape :", x.shape)   # (200, 64, 64, 3)
# print("y.shape :", y.shape)   # (200, 2)

# resize 100 x 100
# # print("x.shape :", x.shape)   # (200, 100, 100, 3)
# # print("y.shape :", y.shape)   # (200, 2)
 
# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)




# # CNN
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


x = np.load('./data/x_data.npy')
y = np.load('./data/y_data.npy')

print("x.shape :", x.shape)
print("y.shape :", y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 11, shuffle = True
)

print("x_train.shape :", x_train.shape)  # (160, 64, 64, 3)
print("x_test.shape :", x_test.shape)    # (40, 64, 64, 3)
print("y_train.shape :", y_train.shape)  # (160, 2)
print("y_test.shape :", y_test.shape)    # (40, 2)


# 데이터 전처리
# scaling(하기 전, 다시 2차원으로 reshape 해 줘야 함)

# 64 x 64
x_train = x_train.reshape(x_train.shape[0], 64*64*3)
x_test = x_test.reshape(x_test.shape[0], 64*64*3)

# # 100 x 100
# # x_train = x_train.reshape(x_train.shape[0], 100*100*3)
# # x_test = x_test.reshape(x_test.shape[0], 100*100*3)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_test)


# CNN 모델에 맞게 reshape
# 64 x 64
x_train = x_train.reshape(x_train.shape[0], 64, 64, 3)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 3)

# # 100 x 100
# # x_train = x_train.reshape(x_train.shape[0] ,100, 100, 3)
# # x_test = x_test.reshape(x_test.shape[0], 100, 100, 3)


print("x_train.shape :", x_train.shape)

# 2. 모델 구성

### 함수형 ###
input1 = Input(shape = (64, 64, 3))
dense1 = Conv2D(90, (2, 2))(input1)
dense2 = Dropout(0.2)(dense1)     
dense3 = Conv2D(100, (3, 3))(dense2)
dense4 = Dropout(0.2)(dense3)     

dense5 = Conv2D(150, (3, 3) , padding = 'same')(dense4)   
dense6 = MaxPooling2D(pool_size = 2)(dense5)
dense7 = Dropout(0.3)(dense6)          

dense8 = Conv2D(30, (2, 2), padding = 'same')(dense7)
dense9 = MaxPooling2D(pool_size = 2)(dense8)
dense10 = Dropout(0.1)(dense9)

dense11 = Flatten()(dense10)
output1 = Dense(2, activation = 'softmax')(dense11)

model = Model(inputs = input1, outputs = output1) 

# # model.summary()

# ### Sequential형 ###
# # model = Sequential()

# # model.add(Conv2D(50, (2, 2), input_shape = (64, 64, 3)))
# # model.add(Conv2D(70, (2, 2), padding = 'same'))
# # model.add(Dense(90))
# # model.add(Dropout(0.3))
# # model.add(MaxPooling2D(pool_size = 2))
# # model.add(Dense(100))
# # model.add(Dropout(0.3))
# # model.add(Dense(30))
# # model.add(MaxPooling2D(pool_size = 2))
# # model.add(Dense(20))
# # model.add(Dense(10))
# # model.add(Flatten())
# # model.add(Dense(2, activation = 'softmax'))

# # model.summary()



# # 3. 컴파일, 훈련
# es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
# modelpath = './model/{epoch:02d}--{acc:.4f}.hdf5'
# cp = ModelCheckpoint(filepath = modelpath, monitor = 'acc', save_best_only = True, mode = 'auto')
# # tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_image = True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split = 0.3, verbose = 1)

# # es + cp
# model.fit(x_train, y_train, epochs = 300, batch_size = 10, validation_split = 0.3,verbose = 1, 
#                                                            callbacks = [es, cp])

# # tb_hist
# # hist = model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split = 0.2, verbose = 1
# #                                    callbacks = [tb_hist] )


# # 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 10)

print("loss :", loss)
print("acc :", acc)

y_pred = model.predict(x_test)

print(np.argmax(y_pred, axis = 1))
# print(y_pred.shape)