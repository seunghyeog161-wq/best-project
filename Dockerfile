FROM python:3.11-slim

# 컨테이너 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app 폴더만 복사 (이게 핵심!)
COPY ./app ./app

# 나머지 파일도 필요하다면 복사
COPY ./README.md ./README.md

EXPOSE 10000

# FastAPI 실행 (main.py 경로와 FastAPI 객체명이 정확)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]

