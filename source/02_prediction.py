from PIL import Image
import numpy as np
import glob

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


# 저장한 모델 사용
from keras.models import load_model
model = load_model('./model/check/check--22--0.4833.hdf5')


# 최종 예측
prediction = model.predict(x_pred)
prediction = np.argmax(prediction, axis = 1)

for i in prediction :
    if i == 0 :
        print("눈을 감고 있습니다zZ")
        print("-------------------")
    else :
        print("눈빛이 살아 있습니다")
        print("-------------------")
