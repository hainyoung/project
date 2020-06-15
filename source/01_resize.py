from PIL import Image
import glob
from natsort import natsorted
# natsorted : to maintatin the order of my filenames
# 만들어놓은 폴더 내 파일 순서로 정렬된다

# display image characteristics
# 사용할 이미지 특성 파악
imc_path = './img/close/ce1.jpg'

imc = Image.open(imc_path)
print('{}'.format(imc.format))
print('size : {}'.format(imc.size))
print('image mode : {}'.format(imc.mode))

# imc.show()

'''
size : (852, 480)
image mode : RGB
'''

# close_eyes image resize

# empty lists
close_list = []     # 사용할 이미지들을 담아 둘 빈 list 생성
resized_close = []  # resize 한 후 이미지들을 담아 둘 빈 list 생성 

# append images to list
for filename in natsorted(glob.glob('./img/close/*.jpg')) : # globe.glob : 특정 폴더의 특정 형식의 파일들을 불러온다(파일명 + 파일경로까지)
    print(filename)
    imc = Image.open(filename) # Image.open : 기존 이미지 파일 열 때 사용
    close_list.append(imc)     # 기존 이미지 파일 imc를 close_list라는 변수(빈 리스트)에 append 하나하나 덧붙여준다

# append resized images to list
for imc in close_list :        # close_list에 있는 이미지들 for문 실행으로 resize 실행
    imc = imc.resize((64, 64)) # 64 x 64로 resize
    resized_close.append(imc)  # resize_close라는 빈 리스트에 resize 된 이미지들(imc)를 append 해 준다
    print('size : {}'.format(imc.size))

# save resized images to new folder
for (i, new) in enumerate(resized_close) :
    new.save ('{}{}{}'.format('./eyes/c_eyes/ce', i+1, '.jpg')) 
# resize 된 이미지들의 리스트



# enumerate 함수
# 리스트가 있는 경우, 순서와 리스트의 값을 전달하는 기능을 가짐
# enumearte : 열거하다
# 이 함수는 순서가 있는 자료형(list, set, tuple, dictionary, string)을
# 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다



'''
# open_eyes image resize

imo_path = './img/open/oe1.jpg'

imo = Image.open(imo_path)
print('{}'.format(imo.format))
print('size : {}'.format(imo.size))
print('image mode : {}'.format(imo.mode))

imo.show()

'''

'''
JPEG
size : (299, 168)
image mode : RGB
'''

'''
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
'''