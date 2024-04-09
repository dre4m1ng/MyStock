import pymongo
from pymongo import MongoClient
import yaml
import pandas as pd
import pprint
import json
import pymysql
import time

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
db_host = _cfg['host']
password = _cfg['password']

# 몽고 db설정
client = pymongo.MongoClient(MONGO_client)
db = client["root"]  # db이름
users_collection = db["stock"]  # 폴더이름

query = {
    "output1.hts_kor_isnm": "삼성전자",
    "output2.stck_bsop_date": {"$exists": True}
}

cursor = users_collection.find(query)

# 결과를 데이터프레임으로 변환할 리스트 초기화
df_list = []
# 중복되서 지워야되는 컬럼
drop_columns = ['prdy_vrss_sign','prdy_vrss', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'acml_vol', 'acml_tr_pbmn']

for data in cursor:
    # output1과 output2를 각각 데이터프레임으로 변환
    output1_df = pd.DataFrame(data['output1'], index=[0])
    output1_df.drop(drop_columns, axis=1, inplace=True)
    output2_df = pd.DataFrame(data['output2'])
    
    # 기존 데이터와 output1, output2 데이터를 합치고 필요없는 열을 제거하여 데이터프레임 생성
    result_df = pd.concat([pd.DataFrame(data, index=[0]), output1_df, output2_df], axis=1)
    result_df = result_df.drop(columns=['output1', 'output2','msg_cd','msg1','rt_cd','_id'])
    
    # df_list에 추가
    df_list.append(result_df)

# 데이터프레임으로 변환
final_df = pd.concat(df_list, ignore_index=True)
final_df.rename(columns={'itewhol_loan_rmnd_ratem name': 'itewhol_loan_rmnd_ratem'}, inplace=True)

# 적재할 데이터 갯수
insert_num = (len(final_df))

# Mysql 데이터베이스 연결
try:
    conn = pymysql.connect(host=db_host, user='root', password=password, database='Mystock')
    print("MySQL 서버에 연결되었습니다.")
except pymysql.err.OperationalError as e:
    print(f"MySQL 연결 오류: {e}")
    exit()

# 커서 생성
cursor = conn.cursor()

# 테이블 이름 설정
table_name = "stock_data"

# 데이터프레임 열 이름을 컬럼 리스트로 변환
columns = ', '.join(final_df.columns)

# 데이터 넣기 전 id 갯수
star_num = cursor.execute("SELECT id FROM stock_data")

# 데이터 삽입
for index, row in final_df.iterrows():
    # VALUES 부분을 동적으로 생성
    values = ', '.join(['%s'] * len(row))

    # SQL 쿼리 생성
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"

    # 행의 값들을 튜플로 변환하여 쿼리 실행
    cursor.execute(sql, tuple(row))
    
    time.sleep(0.2)

# 변경 내용 커밋
conn.commit()

# 데이터 넣은 후 id 갯수
end_num = cursor.execute("SELECT id FROM stock_data")

# 쿼리 성공 여부 확인
if cursor.rowcount > 0:
    print("데이터 삽입 완료.")
else:
    print("데이터 삽입 실패.")

# 커서 및 연결 종료
cursor.close()
conn.close()

print("MySQL 연결 종료.")

# 검수

if insert_num == (end_num-star_num):
    print("무결성 확인")
elif insert_num > (end_num-star_num):
    print("데이터 부족")
else:
    print("데이터 중복")