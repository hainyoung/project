# 이미지 크롤링
# 특정 검색으로 검색한 url을 가지고 나온 이미지들 중에서
# 이미지 주소들의 규칙을 찾고 이런 이미지 주소들을 모아서
# 03번에서 검색어를 텍스트 파일로 저장한 것처럼 이미지 파일을 저장하면 된다

# 이러한 작업을 누군가가 미리 만들어뒀다
# 라이브러리를 이용하면 된다!

# google : python google image search and download
# -> PyPI (현재 막힌 상태)
# 구글에서 제공하는 이미지 크롤링 라이브러리
# pip install google_images_download
# example code 사용

'''
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images
'''
# 이미지 저장이 되지 않음
# pip uninstall google_images_download 후
# https://tiktikeuro.tistory.com/174 블로그 참조하여 크롤링 시도

# 기존 라이브러리를 수정하여 깃허브에 올려주심 능력자!!!!!
# pip install git+https://github.com/Joeclinton1/google-images-download.git

# 키워드당 최대 100개의 사진까지 다운로드
# 아래 코드는 google-images-download 디렉토리에 넣어두고 실행해야 한다

# from google_images_download import google_images_download

# response = google_images_download()

# arguments = {"keywords":"Polar bears, baloons, Beaches", "limit":20, "print_urls":True}
# paths = response.download(arguments)
# print(paths)

# 해 봤는데 안 됨
# importerror 발생 -> 추후 수정 해 보고 다른 방법 찾자

# 네이버 이미지 크롤링

# from urllib.request import urlopen
# from bs4 import BeautifulSoup as bs
# from urllib.parse import quote_plus

# baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
# plusUrl = input('검색어를 입력하세요 : ')
# url = baseUrl + quote_plus(plusUrl)

# html = urlopen(url).read()
# soup = bs(html, "html.parser")
# # img = soup.find_all(class_='_img')
# img = soup.find_all("a", limit = 2) # 개수 제한

# print(img[0])

# n = 1
# for i in img:
#     imgUrl = i['data-source']
#     with urlopen(imgUrl) as f:
#         with open(plusUrl+str(n) + '.jpg', 'wb') as h:
#             img = f.read()
#             h.write(img)
#     n += 1

# print('다운로드완료')

from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
import time
import datetime



baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' 
plusUrl = input('검색어를 입력하세요 : ') # query 뒤에 오는, 크롤링할 이미지 대상의 이름을 기입

start = time.time()

# 한글 검색 자동 변환?
url = baseUrl + quote_plus(plusUrl) 
# 최종으로 크롤링할 url
# quote_plus() : URL로 이동하기 위한 쿼리 문자열을 만들 때 HTMl 폼값을 인용하는 데 필요한 대로 스페이스를 더하기 부호로 치환하기도 합니다. 
# safe에 포함되지 않으면 원래 문자열의 더하기 부호가 이스케이프 됩니다. 또한 safe의 기본값은 '/'가 아닙니다.
# 예: quote_plus('/El Niño/')는 '%2FEl+Ni%C3%B1o%2F'를 산출합니다.

html = urlopen(url).read()
soup = bs(html, "html.parser") # html parsing / html 구문 분석, soup 변수에 넣어둠
img = soup.find_all(class_='_img') # soup에서 class가 image인 부분을 모두 가져옴

print(img[0])

'''
image 를 가져올 수 있는 소스 확인

검색어를 입력하세요 : 황사
<img alt="3M 황사마스크 KF80 황사마스크 구입 | 블로그" class="_img" data-height="325" data-source="https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2F20160122_135%2Fjhmungu_1453445435689KNBJ1_JPEG%2Fhuffpost_com_20160122_154930.jpg&amp;type=b400" data-width="650" onerror="var we=$Element(this); we.addClass('bg_nimg'); we.attr('alt','이미지준비중'); we.attr('src','data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7');" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"/>
PS D:\miniproject> 

'''

# data-source 부분이 이미지 주소에 해당

n = 1
for i in img:
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f: # f = urlopen(imgUrl)
        with open(plusUrl+str(n) + '.jpg', 'wb') as h: # h = 이미지를 저장할 변수
            img = f.read() # imgUrl 읽어옴 / class _img인 부분을 가져온 게 img, 
            h.write(img)
    n += 1

print('다운로드완료')

sec = time.time() - start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print(times)

# 위의 with as 구문을 풀어쓰면 아래와 같다
'''
n = 1
for i in img:
    imgUrl = i['data-source']
    f = urlopen(imgUrl)
    h = open(plusUrl + str(n) + '.jpg', 'wb')
    img = f.read()
    h.write(img)
    n += 1

print("다운로드 완료")
'''



'''
크롤링 결과 검색어명 폴더에 자동 저장

from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
import os

baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
plusUrl = input('검색어를 입력하세요 : ')
# 한글 검색 자동 변환
url = baseUrl + quote_plus(plusUrl)
html = urlopen(url)
soup = bs(html, "html.parser")
img = soup.find_all(class_='_img')

#폴더를 검색어로 생성
dir_path = './img/'
dir_name = plusUrl
os.mkdir(dir_path + "/" + dir_name + "/")
path = dir_path + '/' + dir_name + '/'
n = 1
for i in img:
imgUrl = i['data-source']
with urlopen(imgUrl) as f:
with open(path +plusUrl+str(n)+'.jpg','wb') as h: # w - write b - binary
img = f.read()
h.write(img)
n += 1
print('다운로드 완료')
'''



'''
# 여러 페이지 크롤링

import urllib.request
from bs4 import BeautifulSoup

#접근할 페이지 번호
pageNum = 1

#저장할 이미지 경로 및 이름 (data폴더에 face0.jpg 형식으로 저장)
imageNum = 0
imageStr = "data/face"

while pageNum < 3:
    url = "https://www.kr.playblackdesert.com/BeautyAlbum?searchType=0&searchText=&categoryCode=0&classType=0,4,8,12,16,20,21,24,25,26,28,31,27,19,23,11,29,17,5&Page="
    url = url + str(pageNum)
    
    fp = urllib.request.urlopen(url)
    source = fp.read();
    fp.close()

    soup = BeautifulSoup(source, 'html.parser')
    soup = soup.findAll("p",class_ = "img_area")

    #이미지 경로를 받아 로컬에 저장한다.
    for i in soup:
        imageNum += 1
        imgURL = i.find("img")["src"]
        urllib.request.urlretrieve(imgURL,imageStr + str(imageNum) + ".jpg")
        print(imgURL)
        print(imageNum)

    pageNum += 1
[출처] [python]웹사이트의 대량의 이미지 크롤링하기(2) / 파이썬 웹 크롤러|작성자 유알


'''