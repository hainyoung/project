import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

x = np.load('./data/x_data.npy')
y = np.load('./data/y_data.npy')

x_pred = np.load('./data/x_pred.npy')


print("x.shape :", x.shape)
print("y.shape :", y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 77, shuffle = True
)

print("x_train.shape :", x_train.shape)  # (160, 64, 64, 3)
print("x_test.shape :", x_test.shape)    # (40, 64, 64, 3)
print("y_train.shape :", y_train.shape)  # (160, 2)
print("y_test.shape :", y_test.shape)    # (40, 2)






# Scaler 사용하기 위해 Reshape
x_train = x_train.reshape(x_train.shape[0], 64*64*3)
x_test = x_test.reshape(x_test.shape[0], 64*64*3)

# MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# CNN 모델에 맞게 Reshape
x_train = x_train.reshape(x_train.shape[0], 64, 64, 3)
x_test = x_test.reshape(x_test.shape[0], 64, 64, 3)









# 2. 모델 구성

model = Sequential()

model.add(Conv2D(32, (2, 2), activation = 'relu', input_shape = (64, 64, 3))) 
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))

model.add(Conv2D(64, (2, 2)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Dropout(0.1))
model.add(Dense(16))
model.add(Dropout(0.1))
model.add(Dense(8))
model.add(Dense(2, activation = 'sigmoid'))

model.summary()




# 3. 컴파일, 훈련
# es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')

modelpath = './check/check--{epoch:02d}--{val_loss:.4f}.hdf5'

cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

hist = model.fit(x_train, y_train, epochs = 50, batch_size = 32, validation_split = 0.3, verbose = 1, callbacks = [cp])










# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print("LOSS :", loss)
print("ACC :", acc)

# 시각화

plt.figure(figsize = (12, 10))

plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker = 'o', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = 'o', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('Training and Val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training loss', 'Val loss'], loc = 'upper right')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker = 'o', c = 'purple', label = 'acc' )
plt.plot(hist.history['val_acc'], marker = 'o', c = 'green', label = 'val_loss')
plt.grid()
plt.title('Training and Val accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Training acc', 'Val acc'], loc = 'upper left')

plt.show()








'''
b50
LOSS : 0.7065375208854675
ACC : 0.7749999761581421
'''

'''
b32
LOSS : 0.5462690591812134
ACC : 0.762499988079071
'''

'''
b50
LOSS : 0.6000218272209168
ACC : 0.800000011920929
'''