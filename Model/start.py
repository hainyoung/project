# 데이터 수집
# 데이터 전처리
# 모델 구성
# 컴파일, 훈련
# 평가, 예측

# 이미지 크롤링
# 눈 감은 사진 / 눈 뜬 사진 일단 각 100개까지 수집 후 전처리

# 결과 지표 : ACC
# 어떤 사진을 머신에게 주었을 때
# 눈을 감았는지, 떴는지 -> 이진분류모델


from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus

baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
plusUrl = input('검색어를 입력하세요 : ')
# 한글 검색 자동 변환
url = baseUrl + quote_plus(plusUrl)
html = urlopen(url)
soup = bs(html, "html.parser")
img = soup.find_all(class_='_img')

n = 1
for i in img:
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f:
        with open('./img/' + plusUrl + str(n)+'.jpg','wb') as h: # w - write b - binary
            img = f.read()
            h.write(img)
    n += 1
print('다운로드 완료')

