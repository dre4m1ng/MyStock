import json
import datetime
import time
import yaml
import pandas as pd
import mojito
import pprint
import requests
import pymongo
from pymongo import MongoClient, WriteConcern
from datetime import datetime, timedelta

today = datetime.today().strftime("%Y%m%d")

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

# 2019년 11월 19일부터 시작
start_day = "20200409"
end_day = start_day

# 다음 데이터 추가
while end_day <= today:
    data = []  # 데이터 저장용 리스트 초기화
    print("리셋")

    # 100개 일봉 데이터 요청
    resp = broker.fetch_ohlcv(
        symbol="005930",
        timeframe='D',
        adj_price=True,
        end_day=end_day
    )

    data = resp['output2']  # 응답에서 데이터 추출

    # 다음 100일 날짜 설정
    end_day = (datetime.strptime(end_day, "%Y%m%d") + timedelta(days=100)).strftime("%Y%m%d")
    print(end_day)
    
    # MongoDB에 적재
    client = pymongo.MongoClient(MONGO_client)
    db = client["mystock"]  # db이름
    users_collection = db["price"]  # 폴더이름

    # 데이터 삽입
    insert_result = users_collection.insert_many(data)
    time.sleep(2)

    while True:
        # 삽입 결과 확인
        if insert_result.acknowledged:
            print("Data inserted successfully.")
            break  # 삽입이 성공하면 루프 종료
        else:
            print("Failed to insert data. Waiting for 10 seconds...")
            time.sleep(10)  # 10초 대기
            continue  # 삽입이 실패하면 다시 시도
