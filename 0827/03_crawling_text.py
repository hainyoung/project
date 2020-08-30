# 네이버 실시간 검색순위 크롤링

# 네이버 전체 사이트에서 실시간 검색어 부분만 골라서 텍스트를 가져와야 하는데
# 해당 부분이 어떻게 생겼는지, 특성을 파악해서 선택해 가져와야 한다



# 조코딩 영상
# class가 "ah_k" 였는데 현재 보니까 keyword로 되어 있어서
# 바꿔서 실행해봤는데 아무 것도 출력 되지 않음

# from bs4 import BeautifulSoup
# from urllib.request import urlopen

# response = urlopen('https://www.naver.com')
# soup = BeautifulSoup(response, 'html.parser')

# i = 1 # 검색어의 순위까지 함께 출력하기 위해 i 변수 선언
# for anchor in soup.select("span.keyword"): # span 태그 중 class가 keyword인 것을 가져와라
#   print(str(i) + "위 : " + anchor.get_text()) # anchor의 텍스트, 검색어 내용만 가져 옴
#   i += 1

# select 문법
# select는 태그 이름으로도 가져올 수 있고 몇 번째 태그 이런 식으로 지정해서 가져올 수도 있다
# e.g.
# select("body a") : <body> 태그 안 쪽 자손(자식, 자식의 자식 포함) 중 <a> 태그를 선택하라는 css 문법


# 현재 '조코딩' 강의에서 설명하는 방법은 먹히지 않음
# 댓글에서 https://code-nen.tistory.com/111 사이트 추천

'''
import requests

json = requests.get("https://www.naver.com/srchrank?frm=main").json()
# print(json)

# json에서 data의 항목을 가져와서 ranks 변수에 넣어둔다
ranks = json.get("data")
# print(ranks)

# rank에서 하나씩 꺼내온다, r로
# rank 하나하나에서 "keyword"에 해당하는 값들을 가져와서 keyword라는 변수에 둔다
# keyword == 검색어 내용
# print문으로 출력

i = 1
for r in ranks :
    keyword = r.get("keyword")
    print(str(i) + "위 : " + keyword)
    i += 1

# 검색어 순위를 읽어와서 텍스트 파일로 저장해보자

i = 1
f = open("./0827/newfile.txt", 'w', encoding = 'utf-8')
for r in ranks :
    data = str(i) + "위 : " + r.get("keyword") + "\n"
    i += 1
    f.write(data)
f.close()
'''



'''
콰이어트 플레이스
배슬기
코로나 라이브
시무7조
딩고
태풍 마이삭
심리섭
곰돌이 푸 다시 만나 행복해
포그바
파월
꼰대라떼
007 골든 아이
광주 3단계
기면증
진인 조은산
신소율 남편
목요일 예능
임영웅 사랑의콜센타
계단말고 엘리베이터
조은산
'''

# https://www.naver.com/srchrank?frm=main 이 url에 접속하면
# 다음과 같이 내용을 볼 수 있다
# 

# {"ts":"2020-08-27T23:10:00+0900","sm":"agallgrpmamsi0en0sp0","rop":[{"ag":"all"},{"gr":"01"},{"ma":"-2"},{"si":"00"},{"en":"00"},{"sp":"00"}],
# "data":[{"rank":1,"keyword":"콰이어트 플레이스","keyword_synonyms":["콰이어트 플레이스2"]},
# {"rank":2,"keyword":"배슬기","keyword_synonyms":[]},
# {"rank":3,"keyword":"코로나 라이브","keyword_synonyms":[]},
# {"rank":4,"keyword":"시무7조","keyword_synonyms":[]},
# {"rank":5,"keyword":"딩고","keyword_synonyms":[]},
# {"rank":6,"keyword":"태풍 마이삭","keyword_synonyms":[]},
# {"rank":7,"keyword":"심리섭","keyword_synonyms":[]},
# {"rank":8,"keyword":"곰돌이 푸 다시 만나 행복해","keyword_synonyms":[]},
# {"rank":9,"keyword":"포그바","keyword_synonyms":[]},
# {"rank":10,"keyword":"시무7조상소문","keyword_synonyms":[]},
# {"rank":11,"keyword":"파월","keyword_synonyms":[]},
# {"rank":12,"keyword":"꼰대라떼","keyword_synonyms":[]},
# {"rank":13,"keyword":"007 골든 아이","keyword_synonyms":[]},
# {"rank":14,"keyword":"광주 3단계","keyword_synonyms":[]},
# {"rank":15,"keyword":"기면증","keyword_synonyms":[]},
# {"rank":16,"keyword":"진인 조은산","keyword_synonyms":[]},
# {"rank":17,"keyword":"신소율 남편","keyword_synonyms":[]},
# {"rank":18,"keyword":"임영웅 사랑의콜센타","keyword_synonyms":[]},
# {"rank":19,"keyword":"목요일 예능","keyword_synonyms":[]},
# {"rank":20,"keyword":"계단말고 엘리베이터","keyword_synonyms":[]}]}

import requests 
from bs4 import BeautifulSoup 
from urllib.request import urlopen 

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'} 
url = 'https://datalab.naver.com/keyword/realtimeList.naver?where=main' 
res = requests.get(url, headers = headers) 
soup = BeautifulSoup(res.content, 'html.parser') 
data = soup.select('span.item_title') 
# f = open("./miniproject/0827/rank.txt", 'w')

f = open("./0827/newfile2.txt", 'w', encoding = 'utf-8')
i = 1

for item in data:
    data = "%d위 : "%i + item.get_text() + "\n"
    i = i + 1
    f.write(data)
f.close()