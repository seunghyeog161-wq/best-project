# app/data/db.py
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Base는 models.py에서 import
from .models import Base  

# __file__ 기준으로 data 폴더 안에 app.db 생성 (절대경로)
BASE_DIR = Path(__file__).resolve().parent  # .../app/data
DB_FILE = BASE_DIR / "app.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

# data 폴더가 없으면 생성 (안전)
BASE_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 옵션
    echo=True,  # 쿼리 로그 확인용
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    """앱 시작할 때 한 번 호출해서 테이블 생성"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """요청마다 DB 세션 열고 닫는 의존성"""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
