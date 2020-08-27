# beautiful soup 사용
# html, xml 구문을 분석하기 위한 파이썬 패키지(파이썬 라이브러리)


# beautiful soup wikipedia
# 위키피디아 예제 무작정 따라하기
# Code example copy&paste

#!/usr/bin/env python3
# Anchor extraction from HTML document

# import library
from bs4 import BeautifulSoup
from urllib.request import urlopen

# urlopen : 페이지를 열어준다, response라는 곳으로 담겠다
with urlopen('https://en.wikipedia.org/wiki/Main_Page') as response:
    soup = BeautifulSoup(response, 'html.parser')
    for anchor in soup.find_all('a'):
        print(anchor.get('href', '/'))

# with ~ as ~ : 파이썬 문법
# with as 구문은 다음과 같이 직관적으로 쓸 수도 있다
# responose = urlopen('https://en.wikipedia.org/wiki/Main_Page') 
# soup = BeautifulSoup(response, 'html.parser')
# for anchor in soup.find_all('a'):
#     print(anchor.get('href', '/'))

# BeautifulSoup() 함수를 사용, response를 넣어주고 html.parser를 이용하여 구문을 분석한 것을
# soup 이라는 변수에 담아준다

# for문을 사용하여 soup 안에 있는 'a' Tag를 찾아서
# anchor 라는 변수에 넣는다
# for문을 통해 하나씩 가져온 anchor 안에 a 태그의 'href' 참조 주소를 가져와서 print 

