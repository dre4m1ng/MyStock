import json
import datetime
import time
import yaml
import pandas as pd
import mojito
import pprint
import requests
import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta

with open('C:/Users/cc843/Desktop/辛旻宗/주식프로젝트/key/config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
ACCESS_TOKEN = ""
CANO = _cfg['CANO']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']
DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']
URL_BASE = _cfg['URL_BASE']
MONGO_client = _cfg['mongodb']

broker = mojito.KoreaInvestment(
    api_key=APP_KEY,
    api_secret=APP_SECRET,
    acc_no="50101465-01",
    mock=True
)

# 오늘
today = datetime.today().strftime("%Y%m%d")

# 2019년 11월 19일부터 시작
day = "20190101"

# 다음 데이터 추가
while day <= today:
    data = []  # 데이터 저장용 리스트 초기화
    print("리셋")

    while True:
        # 1개 일봉 데이터 요청
        resp = broker.fetch_ohlcv(
            symbol="005930",
            timeframe='D',
            adj_price=True,
            start_day=day,
            end_day=day
        )
        time.sleep(0.5)
        if resp['msg1'] == "초당 거래건수를 초과하였습니다.":
            print("초당 거래량 초과! 대기")
            time.sleep(0.3)
        else:
            break
    # 다음날 설정
    day = (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    print(day)

    # MongoDB 설정
    client = pymongo.MongoClient(MONGO_client)
    db = client["root"]  # db이름
    users_collection = db["stock"]  # 폴더이름

    # 넣을 데이터
    data = resp

    # 데이터 삽입
    insert_result = users_collection.insert_one(data)
    
    # 삽입 후 잠시 대기
    time.sleep(0.2)
    while True:
        # 삽입 결과 확인
        if insert_result.acknowledged:
            print("Data inserted successfully.")
            break  # 삽입이 성공하면 루프 종료
        else:
            print("Failed to insert data. Waiting for 10 seconds...")
            time.sleep(0.2)  # 0.2초 대기
            continue  # 삽입이 실패하면 다시 시도
