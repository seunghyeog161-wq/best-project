FROM python:3.11-slim

# 컨테이너 작업 디렉토리
WORKDIR /app

# pip 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 프로젝트 복사
COPY . .

# FastAPI 실행
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:10000", "--timeout", "120"]
