from datetime import datetime
import time
import yaml
import pandas as pd
import mojito
import requests
import pymongo
from pymongo import MongoClient

import os
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("APP_KEY")
APP_SECRET = os.getenv("APP_SECRET")
ACCESS_TOKEN = ""
# CANO = _cfg['CANO']
CANO = os.getenv("CANO")
# ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']
ACNT_PRDT_CD = os.getenv("ACNT_PRDT_CD")
# DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']
URL_BASE = os.getenv('URL_BASE')

MONGO_client = os.getenv("MONGODB_URI")
MONGO_DB = os.getenv("MONGODB_DB")

broker = mojito.KoreaInvestment(
    api_key=APP_KEY,
    api_secret=APP_SECRET,
    acc_no="50101465-01",
    mock=True
)

def fetch_and_insert_data(symbol="005930"): # 종목

    # 오늘
    today = datetime.today().strftime("%Y%m%d")

    # 데이터 저장용 리스트 초기화
    data = []

    # 1개 일봉 데이터 요청
    while True:
        resp = broker.fetch_ohlcv(
            symbol=symbol,
            timeframe='D',
            adj_price=True,
            start_day=today,
            end_day=today
        )
        time.sleep(0.4)

        if resp['msg1'] == "초당 거래건수를 초과하였습니다.":
            print("초당 거래량 초과! 대기")
            time.sleep(0.2)
        else:
            break

    # MongoDB 설정
    client = pymongo.MongoClient(MONGO_client)
    db = client["root"]  # db이름
    users_collection = db["test"]  # 폴더이름

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
    print(f'{symbol}insert 완료')    

top_stock = ['097950', '383220', '002310', '051910', '207940', '003670','005490',
             '034020', '005930', '137310', '005380', '028260', '015760', '000720',
             '011200', '017670', '105560', '006800', '032830', '035420']

if __name__ == "__main__":
    # 1위 종목들 데이터 추가
    for top in top_stock:
        fetch_and_insert_data(symbol=top)