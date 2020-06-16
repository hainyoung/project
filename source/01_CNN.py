from PIL import Image
import glob
from natsort import natsorted
import os
import cv2
import numpy as np

#### 이미지 리사이즈
# 1) closed eyes resize
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


#### NUMPY conversion
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



# 최종 예측 위한 데이터 준비
testimg_dir = './img/test/'
image_w = 64
image_h = 64

x_pred = []
imgname = []

testimg = glob.glob(testimg_dir + '*.jpg')

for i, f in enumerate(testimg) :
    image = Image.open(f)
    image = image.convert("RGB")
    image = image.resize((image_w, image_h))
    data = np.asarray(image, dtype = 'float32')
    imgname.append(image)
    x_pred.append(data)

x_pred = np.array(x_pred)


# numpy로 최종 저장
np.save('./data/x_data.npy', x)
np.save('./data/y_data.npy', y)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

x = np.load('./data/x_data.npy')
y = np.load('./data/y_data.npy')

print("x.shape :", x.shape)
print("y.shape :", y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 77, shuffle = True
)

print("x_train.shape :", x_train.shape)  # (160, 64, 64, 3)
print("x_test.shape :", x_test.shape)    # (40, 64, 64, 3)
print("y_train.shape :", y_train.shape)  # (160, 2)
print("y_test.shape :", y_test.shape)    # (40, 2)


x_train = x_train.astype(float)/255
x_test = x_test.astype(float)/255

print("x_train.shape :", x_train.shape)


# 2. 모델 구성

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3))) 
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Dense(2, activation = 'sigmoid'))
model.summary()



# 3. 컴파일, 훈련
es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
modelpath = './model/{epoch:02d}--{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')
# tb_hist = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_image = True)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs = 32, batch_size = 32, validation_split = 0.3, verbose = 1, callbacks = [cp])
                                  

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print("LOSS :", loss)
print("ACC :", acc)

prediction = model.predict(x_pred)
prediction = np.argmax(prediction, axis = 1)

for i in prediction :
    if i == 0 :
        print("눈을 감고 있습니다zZ")
    else :
        print("눈빛이 살아 있습니다")


# 시각화

plt.figure(figsize = (12, 10))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = 'o', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = 'v', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('Training and Val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training loss', 'Val loss'], loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = 'o', c = 'purple', label = 'acc' )
plt.plot(hist.history['val_acc'], marker = 'v', c = 'green', label = 'val_loss')
plt.grid()
plt.title('Training and Val accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Training acc', 'Val acc'], loc = 'upper left')

plt.show()
