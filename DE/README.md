# TODO
- [x] Dockerizing
- [x] DB 스키마 설계
- [x] 한국투자증권 api 사용하여 일봉 데이터 가져오는 코드 작성
- [x] 데이터 적재
- [x] jenkins 빌드 테스트 job 등록
- [ ] DS분들이 작성한 모델 탑재
- [ ] 시각화(Grafana)
- [ ] Airflow로 데이터 파이프라인 자동화
- [ ] 모델 백테스트 기능 추가
- [ ] 매수, 매도 기능 추가
- [ ] 디스코드 알림 기능 추가

# 코드 설명
```bash
mojito 모듈을 사용한 주식 데이터 적재
https://pypi.org/project/mojito2/
한국투자증권 api로 받아온 response를 MongoDB에 적재
이후 필요한 데이터들을 분류하여 MySQL에 적재
```

---
# 구축 메뉴얼
## Architecture
![](MyStock-architecture.png)
## Install Components

### Airflow
```bash
helm repo add apache-airflow https://airflow.apache.org/
helm install airflow apache-airflow/airflow
```
### Requirements
```bash
pip install -r requirements.txt
```

### Docker Images
```bash
docker pull mysql:latest
docker pull jupyter/scipy-notebook:latest
docker pull mongo:latest
docker pull jenkins/jenkins:lts-jdk17
docker pull apache/airflow:2.9.0
```

### Jenkins 사용 방법
- Jenkins 컨테이너에 python 3.11.8 설치
- Requirements 설치
- Jenkins 컨테이너 내부에 .env파일 작성 이후 Jenkins Job에 .env 경로 복사 쉘스크립트 추가