#!/usr/bin/env python
# -*- coding: utf-8 -*-

from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup
import requests
import os



# 크롤링 옵션 생성
options = webdriver.ChromeOptions()
# 백그라운드 실행 옵션 추가
#options.add_argument("headless")

# 웹 드라이버 실행
driver = webdriver.Chrome(r'C:\\<<PATH>>\\chromedriver.exe', chrome_options=options)
driver.get('https://e.kakao.com/t/<<URL>>')

# HTML 소스코드 가져오기
html = driver.page_source 



# HTML 소스코드 출력
#print(html.encode('utf-8'))
#sleep(5)
# BeautifulSoup으로 HTML 파싱
soup = BeautifulSoup(html, 'html.parser')

# img_emoticon 클래스를 가진 img 요소 찾기
img_elements = soup.select('img.img_emoticon')

# 이미지 다운로드할 폴더 생성
if not os.path.exists('images'):
    os.makedirs('images')

# img 요소의 src 가져오기
for idx, img in enumerate(img_elements):
    img_url = img.get('src')
    if img_url:
        response = requests.get(img_url)
        with open('images/image_{}.png'.format(idx), 'wb') as f:
            f.write(response.content)

# 드라이버 종료
driver.quit()
