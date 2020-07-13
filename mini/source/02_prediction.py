
from PIL import Image
import numpy as np
import glob
from keras.models import load_model

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
    image.save ('{}{}{}'.format('./eyes/test/test', i+1, '.jpg'))
    data = np.asarray(image, dtype = 'float32')
    # print(data)
    imgname.append(image)
    x_pred.append(data)

x_pred = np.array(x_pred)

# 저장한 모델 사용
model = load_model('./check/save/check--32--0.1243.hdf5')

# 최종 예측
prediction = model.predict(x_pred)

cnt = 0
for i in prediction :
    result = i.argmax()
    if result == 0 : result_str = "눈을 감고 있는 사진"
    else : result_str = "눈을 뜨고 있는 사진"

    if i [0] == 1.0 : print("파일명 " + testimg[cnt].split("\\")[1] + " : 이것은 " + result_str + "으로 보입니다.")
    if i [1] == 1.0 : print("파일명 " + testimg[cnt].split("\\")[1] + " : 이것은 " + result_str + "으로 보입니다.")
    cnt += 1








# split("\\") : \ 기호를 기준으로 문자열을 나눠준다
# print(testimg[0]) = ./img/test\c1.jpg
#                          [0]   [1]
# split("\\")[0] = ./img/test
# split("\\")[1] = c1.jpg

# np.save('./data/x_pred.npy', arr = x_pred)
# x_pred = np.load('./data/x_pred.npy')
