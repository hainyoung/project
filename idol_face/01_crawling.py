from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
import os


# stars = ['안소희', '시우민', '강슬기', '황민현', '박보영', '손예진', '박보검', '박형식', '김우빈', '신민아', '종현', '장재인', '배수지', '전정국', '나연', '강찬희']

sm = ['수호', '강타', '최시원', '김준면', '안칠현', '슈퍼주니어 시원']
yg = ['지드래곤', '송민호', '비아이', '권지용', '위너 송민호', '김한빈']
jyp = ['준호', 'JB', '정지훈', '2pm 준호', '갓세븐 jb', '가수 비']


for plusUrl in sm:
# for plusUrl in yg:
# for plusUrl in jyp:

    baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='

    # 한글 검색 자동 변환
    url = baseUrl + quote_plus(plusUrl)
    html = urlopen(url)
    soup = bs(html, "html.parser")
    img = soup.find_all(class_='_img')

    # 폴더를 검색어로 생성
    dir_path = './idol_face/images/boys'
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

print("complete")
