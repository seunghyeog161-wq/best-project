import httpx
from urllib.parse import quote
import logging

from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path
import logging, os
logging.getLogger("uvicorn.error").warning(f"[BOOT] Loaded from: {__file__}")
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Response, Request, Header, Query

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, FileResponse, PlainTextResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.staticfiles import StaticFiles

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy import (
    create_engine, text, String, Integer, DateTime, Boolean,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import (
    DeclarativeBase, mapped_column, Mapped, relationship,
    sessionmaker, Session
)

from authlib.integrations.starlette_client import OAuth
# http 로 테스트할 때(required)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
if BASE_URL.startswith("http://"):
    os.environ["AUTHLIB_INSECURE_TRANSPORT"] = "1"

# ─────────────────────────────────────────────────────────────────
# 0) 경로/환경
# ─────────────────────────────────────────────────────────────────
BASE_DIR_APP = Path(__file__).resolve().parent
BASE_DIR = BASE_DIR_APP.parent
# BASE_DIR_APP / BASE_DIR 밑에 추가

def _desktop_dir() -> Path:
    home = Path.home()
    candidates = [
        home / "OneDrive" / "바탕 화면",
        home / "OneDrive" / "Desktop",
        home / "바탕 화면",
        home / "Desktop",
        home / "바탕화면",
    ]
    for p in candidates:
        if p.exists():
            return p
    return home  # fallback

DESKTOP_DIR = _desktop_dir()
FAVICON_ICO = DESKTOP_DIR / "favicon.ico"

# app/main.py









def _find_login_html() -> Path | None:
    env_path = os.getenv("FRONTEND_LOGIN_FILE")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    for p in (BASE_DIR / "frontend").rglob("login.html"):
        return p

    home = Path.home()
    candidates = [
        home / "OneDrive" / "바탕 화면" / "login.html",
        home / "OneDrive" / "Desktop" / "login.html",
        home / "바탕 화면" / "login.html",
        home / "Desktop" / "login.html",
        home / "바탕화면" / "login.html",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

LOGIN_HTML = _find_login_html()

# .env는 프로젝트 루트(myproject)에 둔다
load_dotenv(dotenv_path=BASE_DIR / ".env")

APP_ENV = os.getenv("APP_ENV", "dev")
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", BASE_URL)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

JWT_SECRET = os.getenv("JWT_SECRET", "change_this_secret")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 15))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN", "")  # 로컬 개발이면 빈 값 권장
SECURE_COOKIES = os.getenv("SECURE_COOKIES", "false").lower() == "true"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
META_CLIENT_ID = os.getenv("META_CLIENT_ID")          # Facebook/Instagram
META_CLIENT_SECRET = os.getenv("META_CLIENT_SECRET")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-session-secret")

# 바탕화면 정적 자산 임시 서빙 여부(보안상 기본 False)
SERVE_LOGIN_ASSETS = os.getenv("SERVE_LOGIN_ASSETS", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────
# 1) DB & 모델
# ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=(APP_ENV == "dev"),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    social_accounts: Mapped[List["UserSocialAccount"]] = relationship(back_populates="user")

    # 👇 로그인 추적용
    login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)



class UserSocialAccount(Base):
    __tablename__ = "user_social_accounts"
    __table_args__ = (UniqueConstraint("provider", "provider_user_id", name="uq_provider_user"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(String(50))  # google | meta | naver
    provider_user_id: Mapped[str] = mapped_column(String(255))
    access_token: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    refresh_token: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    user: Mapped[User] = relationship(back_populates="social_accounts")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────────────────────────
# 2) 보안/JWT
# ─────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
REFRESH_COOKIE_NAME = "refresh_token"

def hash_password(raw: str) -> str:
    return pwd_context.hash(raw)

def verify_password(raw: str, hashed: str) -> bool:
    return pwd_context.verify(raw, hashed)

def _expire(minutes: int | None = None, days: int | None = None) -> datetime:
    now = datetime.now(timezone.utc)
    if minutes is not None:
        return now + timedelta(minutes=minutes)
    if days is not None:
        return now + timedelta(days=days)
    return now + timedelta(minutes=15)

def create_access_token(sub: str) -> str:
    exp = _expire(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": sub, "exp": exp}, JWT_SECRET, algorithm=JWT_ALG)

def create_refresh_token(sub: str) -> str:
    exp = _expire(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": sub, "exp": exp, "type": "refresh"}, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def set_refresh_cookie(response: Response, token: str):
    cookie_kwargs = dict(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=SECURE_COOKIES,
        samesite="lax",
        path="/",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )
    if COOKIE_DOMAIN:
        cookie_kwargs["domain"] = COOKIE_DOMAIN
    response.set_cookie(**cookie_kwargs)

def clear_refresh_cookie(response: Response):
    if COOKIE_DOMAIN:
        response.delete_cookie(key=REFRESH_COOKIE_NAME, domain=COOKIE_DOMAIN, path="/")
    else:
        response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/")

# ─────────────────────────────────────────────────────────────────
# 3) 스키마
# ─────────────────────────────────────────────────────────────────
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: Optional[str] = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # Pydantic v2
    id: int
    email: EmailStr
    name: Optional[str] = None
    is_admin: bool = False

# ─────────────────────────────────────────────────────────────────
# 4) 앱/미들웨어
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Auth Backend (single file)", debug=(APP_ENV == "dev"))
from app.reco import MEDIA_DIR, router as reco_router

app.include_router(reco_router, prefix="/api/reco")
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_ORIGIN,          # .env에서 지정한 프론트
        BASE_URL,                 # 동일 오리진일 때
        "http://localhost:3000",  # Vite/CRA 개발 서버 흔한 포트
        "http://127.0.0.1:3000",
        "http://localhost:8000",  # 같은 포트에서 정적 서빙할 때
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,                   # 쿠키/세션 전송 허용
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "Accept"],  # 명시적으로 허용
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")

logger = logging.getLogger("uvicorn.error")

@app.middleware("http")
async def log_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception("Unhandled error on %s %s: %s", request.method, request.url, str(e))
        return JSONResponse({"detail": "server error"}, status_code=500)

# 선택적: 바탕화면 정적 자산 임시 서빙(⚠️ 전체 노출 위험, 로컬 개발 때만)
if SERVE_LOGIN_ASSETS:
    logger.warning("[SECURITY] SERVE_LOGIN_ASSETS=True: Exposing Desktop directory at /login-static (dev only)")
    app.mount("/login-static", StaticFiles(directory=str(DESKTOP_DIR)), name="login-static")


# 루트("/") 접속 시 바탕화면의 login.html 반환
# app/main.py — 루트("/") 라우트 교체
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse

@app.get("/", response_class=HTMLResponse)
def serve_login():
    # 1) 바탕화면/프로젝트 등에서 login.html 찾기 (이미 위에서 LOGIN_HTML 계산함)
    if LOGIN_HTML and LOGIN_HTML.exists():
        return FileResponse(
            str(LOGIN_HTML),
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache"}
        )

    # 2) (대안) frontend/index.html이 있으면 거기로 리다이렉트
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return RedirectResponse("/frontend/index.html", status_code=302)

    # 3) 둘 다 없으면 안내
    return HTMLResponse(
        "<h1>login.html을 찾을 수 없습니다</h1>"
        "<p>.env에 <b>FRONTEND_LOGIN_FILE=경로</b> 를 지정하거나,<br>"
        "<code>프로젝트/frontend/</code> 또는 바탕화면에 <b>login.html</b>을 두세요.</p>",
        status_code=404,
    )

# 선택: /login.html로 접근해도 동일 동작 원하면 유지
@app.get("/login.html")
def login_html_alias():
    return RedirectResponse("/", status_code=307)



# favicon이 있으면 제공
@app.get("/favicon.ico")
def favicon():
    if FAVICON_ICO.exists():
        return FileResponse(str(FAVICON_ICO), media_type="image/x-icon")
    return PlainTextResponse("", status_code=204)

# 헬스체크
@app.get("/healthz")
def healthz():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"ok": True}

@app.get("/ping")
def ping():
    return {"ok": True}


from PIL import Image
import io
# reco 라우터 연결 (/api/reco/*)

@app.get("/media/resize")
def media_resize(p: str = Query(...), w: int = 680, h: int = 520, mode: str = "cover"):
    path = (MEDIA_DIR / p).resolve()
    if not str(path).startswith(str(MEDIA_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    im = Image.open(path).convert("RGB")
    if mode == "cover":
        src_w, src_h = im.size
        target_ratio = w / h
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            new_w = int(src_h * target_ratio)
            x1 = (src_w - new_w) // 2
            im = im.crop((x1, 0, x1 + new_w, src_h))
        else:
            new_h = int(src_w / target_ratio)
            y1 = (src_h - new_h) // 2
            im = im.crop((0, y1, src_w, y1 + new_h))
        im = im.resize((w, h), Image.LANCZOS)
    else:
        im.thumbnail((w, h), Image.LANCZOS)

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

# 디버그 라우트
@app.get("/debug/paths")
def debug_paths():
    return {
        "DESKTOP_DIR": str(DESKTOP_DIR),
        "LOGIN_HTML": str(LOGIN_HTML) if LOGIN_HTML else None,
        "exists_LOGIN_HTML": (LOGIN_HTML.exists() if LOGIN_HTML else False),
    }


@app.get("/login.html")
def login_html_alias():
    return RedirectResponse("/", status_code=307)

@app.get("/debug/google")
def debug_google():
    return {
        "redirect_uri_runtime": f"{BASE_URL}/auth/google/callback",
        "session_middleware": True,
    }

# ─────────────────────────────────────────────────────────────────
# 5) 이메일/비번 로그인
# ─────────────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=UserOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    u = User(email=payload.email, password_hash=hash_password(payload.password), name=payload.name)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u

@app.post("/auth/login", response_model=TokenOut)
def login(payload: LoginIn, response: Response, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.email == payload.email).first()
    if not u or not u.password_hash or not verify_password(payload.password, u.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    access = create_access_token(str(u.id))
    refresh = create_refresh_token(str(u.id))
    set_refresh_cookie(response, refresh)
    return TokenOut(access_token=access)

@app.post("/auth/refresh", response_model=TokenOut)
def refresh(request: Request):
    rt = request.cookies.get(REFRESH_COOKIE_NAME)
    if not rt:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    data = decode_token(rt)
    if data.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return TokenOut(access_token=create_access_token(data["sub"]))

@app.get("/auth/me", response_model=UserOut)
def me(authorization: Optional[str] = Header(None, alias="Authorization"), db: Session = Depends(get_db)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing access token")
    data = decode_token(authorization.split(maxsplit=1)[1])
    u = db.get(User, int(data["sub"]))
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u





oauth = OAuth()
def _after_redirect(access_token: str) -> RedirectResponse:
    return RedirectResponse(url=f"/auth/after?access_token={access_token}", status_code=302)

if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

if META_CLIENT_ID and META_CLIENT_SECRET:
    oauth.register(
        name="meta",
        client_id=META_CLIENT_ID,
        client_secret=META_CLIENT_SECRET,
        access_token_url="https://graph.facebook.com/v20.0/oauth/access_token",
        authorize_url="https://www.facebook.com/v20.0/dialog/oauth",
        api_base_url="https://graph.facebook.com/",
        client_kwargs={"scope": "email public_profile"},
    )

if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
    oauth.register(
        name="naver",
        client_id=NAVER_CLIENT_ID,
        client_secret=NAVER_CLIENT_SECRET,
        access_token_url="https://nid.naver.com/oauth2.0/token",
        authorize_url="https://nid.naver.com/oauth2.0/authorize",
        api_base_url="https://openapi.naver.com/v1/nid/",
        client_kwargs={"scope": "name email"},
    )

def _frontend_callback(access_token: str) -> RedirectResponse:
    # index.html?access_token=... 로 이동
    return RedirectResponse(
        url=f"/frontend/index.html?access_token={access_token}",
        status_code=302
    )

# ── Google
@app.get("/auth/google/login")
async def google_login(request: Request):
    # 1) 리프레시 쿠키 있으면 토큰 갱신해서 곧장 계정확인 화면으로
    rt = request.cookies.get(REFRESH_COOKIE_NAME)
    if rt:
        try:
            data = decode_token(rt)
            if data.get("type") == "refresh":
                access = create_access_token(data["sub"])
                return _after_redirect(access)
        except Exception:
            pass  # 쿠키가 깨졌으면 정상 OAuth 진행

    # 2) OAuth 시작 (강제 계정 선택 지원: /auth/google/login?force=true)
    client = oauth.create_client("google")
    if not client:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")

    redirect_uri = str(request.url_for("google_callback"))
    params = {}
    if request.query_params.get("force") == "true":
        # 구글 계정 선택창 무조건 띄우기
        params["prompt"] = "select_account"

    logging.getLogger("uvicorn.error").warning(f"[OAUTH][GOOGLE] redirect_uri={redirect_uri} params={params}")
    return await client.authorize_redirect(request, redirect_uri, **params)


@app.get("/auth/google")
async def google_login_alias(request: Request):
    return await google_login(request)

@app.get("/auth/google/callback")
async def google_callback(request: Request, response: Response, db: Session = Depends(get_db)):
    log = logging.getLogger("uvicorn.error")
    try:
        client = oauth.create_client("google")
        token = await client.authorize_access_token(request)
        log.warning(f"[OAUTH][GOOGLE] token keys = {list(token.keys())}")
        log.warning(f"[OAUTH][GOOGLE] token = {token}")

        # ✅ userinfo 안전하게 추출
        userinfo = token.get("userinfo")
        if not userinfo:
            try:
                userinfo = await client.parse_id_token(request, token)
            except Exception as e:
                log.exception("[OAUTH][GOOGLE] Failed to parse ID token")
                return HTMLResponse("<h3>Google ID 토큰을 파싱할 수 없습니다.</h3>", status_code=500)

        log.warning(f"[OAUTH][GOOGLE] userinfo = {userinfo}")

        provider_user_id = userinfo.get("sub")
        email = userinfo.get("email")
        name = userinfo.get("name", "")

        if not provider_user_id:
            return HTMLResponse("<h3>Google 응답에 'sub' 항목이 없습니다.</h3>", status_code=400)

        # 기존 소셜 계정 확인
        sa = db.query(UserSocialAccount).filter(
            UserSocialAccount.provider == "google",
            UserSocialAccount.provider_user_id == provider_user_id,
        ).first()

        if sa:
            user = db.get(User, sa.user_id)
        else:
            user = db.query(User).filter(User.email == email).first() if email else None

            if not user:
                user = User(
                    email=email or f"google:{provider_user_id}@no-email.local",
                    name=name,
                    is_active=True,
                )
                db.add(user)
                db.commit()
                db.refresh(user)

            sa = UserSocialAccount(
                user_id=user.id,
                provider="google",
                provider_user_id=provider_user_id,
                access_token=token.get("access_token"),
                refresh_token=token.get("refresh_token"),
            )
            db.add(sa)
            db.commit()

        access = create_access_token(str(user.id))
        refresh = create_refresh_token(str(user.id))
        set_refresh_cookie(response, refresh)
        return _after_redirect(access)

    except Exception as e:
        log.exception("[OAUTH][GOOGLE][ERROR]")
        return HTMLResponse(
            "<h3 style='color:red'>Google 로그인 처리 중 오류가 발생했습니다.</h3>"
            "<p>개발자 콘솔 설정, 토큰 구조, Callback URL 등을 다시 확인하세요.</p>"
            f"<pre>{str(e)}</pre>",
            status_code=500,
        )


# ── Facebook (Meta)
@app.get("/auth/facebook/login")
async def facebook_login(request: Request):
    client = oauth.create_client("meta")
    if not client:
        raise HTTPException(status_code=400, detail="Meta OAuth not configured")
    redirect_uri = str(request.url_for("facebook_callback"))
    logging.getLogger("uvicorn.error").warning(f"[OAUTH][META] redirect_uri={redirect_uri}")
    return await client.authorize_redirect(request, redirect_uri)

@app.get("/auth/facebook/callback")
async def facebook_callback(request: Request, response: Response, db: Session = Depends(get_db)):
    client = oauth.create_client("meta")
    token = await client.authorize_access_token(request)
    resp = await client.get("me", params={"fields": "id,name,email"}, token=token)
    data = resp.json()

    provider_user_id = data.get("id")
    email = data.get("email")
    name = data.get("name", "")

    sa = db.query(UserSocialAccount).filter(
        UserSocialAccount.provider == "meta",
        UserSocialAccount.provider_user_id == provider_user_id,
    ).first()

    if sa:
        user = db.get(User, sa.user_id)
    else:
        user = db.query(User).filter(User.email == email).first() if email else None
        if not user:
            user = User(email=email or f"meta:{provider_user_id}@no-email.local", name=name, is_active=True)
            db.add(user); db.commit(); db.refresh(user)
        if not sa:
            sa = UserSocialAccount(
                user_id=user.id, provider="meta", provider_user_id=provider_user_id,
                access_token=token.get("access_token"), refresh_token=token.get("refresh_token"),
            )
            db.add(sa); db.commit()

    access = create_access_token(str(user.id))
    refresh = create_refresh_token(str(user.id))
    set_refresh_cookie(response, refresh)
    return _frontend_callback(access)

# ── Naver
@app.get("/auth/naver/login")
async def naver_login(request: Request):
    # 1) refresh 쿠키 있으면 곧장 계정확인 화면으로
    rt = request.cookies.get(REFRESH_COOKIE_NAME)
    if rt:
        try:
            data = decode_token(rt)
            if data.get("type") == "refresh":
                access = create_access_token(data["sub"])
                return RedirectResponse(f"/auth/after?access_token={access}", status_code=302)
        except Exception:
            pass

    client = oauth.create_client("naver")
    if not client:
        raise HTTPException(status_code=400, detail="Naver OAuth not configured")

    # ✅ 반드시 request.url_for 사용 (문자열 덧붙이지 않기)
    redirect_uri = str(request.url_for("naver_callback"))
    logging.getLogger("uvicorn.error").warning(f"[OAUTH][NAVER][STEP1] redirect_uri={redirect_uri}")

    # 네이버는 prompt 같은 강제 계정선택 파라미터 없음
    return await client.authorize_redirect(request, redirect_uri)


@app.get("/auth/naver/callback")
async def naver_callback(request: Request, response: Response, db: Session = Depends(get_db)):
    client = oauth.create_client("naver")
    log = logging.getLogger("uvicorn.error")

    try:
        # 1) 토큰 교환
        token = await client.authorize_access_token(request)
        log.warning(f"[OAUTH][NAVER][STEP2] token_keys={list(token.keys())}")

        # 2) 프로필 조회 (v1/nid/me)
        resp = await client.get("me", token=token)
        raw = await resp.aread()
        log.warning(f"[OAUTH][NAVER][STEP3] profile_status={resp.status_code} body={raw.decode(errors='ignore')}")
        data = resp.json().get("response", {})

        provider_user_id = data.get("id")
        email = data.get("email")
        name = data.get("name", "")

        if not provider_user_id:
            # 콘솔/동의항목 문제일 때 사용자에게 힌트 표시
            return HTMLResponse(
                "<h3 style='color:red'>네이버 응답에 id가 없습니다. 개발자센터의 '네이버 아이디로 로그인' 설정과 Callback URL을 확인하세요.</h3>",
                status_code=400,
            )

        # 3) 계정 연결/생성
        sa = db.query(UserSocialAccount).filter(
            UserSocialAccount.provider == "naver",
            UserSocialAccount.provider_user_id == provider_user_id,
        ).first()

        if sa:
            user = db.get(User, sa.user_id)
        else:
            # ✅ 이메일이 있으면 기존 유저 찾기
            user = db.query(User).filter(User.email == email).first() if email else None

            if not user:
                # ✅ 기존 유저가 없을 때만 새로 생성
                user = User(
                    email=email or f"naver:{provider_user_id}@no-email.local",
                    name=name,
                    is_active=True
                )
                db.add(user)
                db.commit()
                db.refresh(user)

            # ✅ 소셜 계정이 없으므로 연결
            sa = UserSocialAccount(
                user_id=user.id,
                provider="naver",
                provider_user_id=provider_user_id,
                access_token=token.get("access_token"),
                refresh_token=token.get("refresh_token"),
            )
            db.add(sa)
            db.commit()


        # 4) 토큰 세팅 + after로 이동
        access = create_access_token(str(user.id))
        refresh = create_refresh_token(str(user.id))
        set_refresh_cookie(response, refresh)
        return RedirectResponse(f"/auth/after?access_token={access}", status_code=302)

    except Exception as e:
        log.exception(f"[OAUTH][NAVER][ERROR] {e}")
        return HTMLResponse(
            "<h3 style='color:red'>네이버 로그인 처리 중 오류가 발생했습니다.</h3>"
            "<p>개발자 로그(uvicorn.error)를 확인하세요. (Callback URL / 동의항목 / 서비스 URL 불일치 가능성)</p>",
            status_code=500,
        )

@app.post("/auth/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    # 1) refresh 토큰으로 현재 사용자 확인
    rt = request.cookies.get(REFRESH_COOKIE_NAME)
    user = None
    if rt:
        try:
            data = decode_token(rt)
            if data.get("type") == "refresh":
                user_id = int(data["sub"])
                user = db.get(User, user_id)
        except Exception:
            user = None

    # 2) 구글 토큰 revoke (있을 때만, 에러는 무시하고 진행)
    if user:
        try:
            sa_google = (
                db.query(UserSocialAccount)
                .filter(
                    UserSocialAccount.user_id == user.id,
                    UserSocialAccount.provider == "google",
                )
                .first()
            )
            if sa_google:
                # 구글 권장 revoke 엔드포인트
                revoke_url = "https://oauth2.googleapis.com/revoke"
                async def _revoke(token: str):
                    async with httpx.AsyncClient(timeout=5) as client:
                        await client.post(revoke_url, data={"token": token})

                # access / refresh 둘 다 가능하면 반납
                import anyio
                tasks = []
                if sa_google.access_token:
                    tasks.append(_revoke(sa_google.access_token))
                if sa_google.refresh_token:
                    tasks.append(_revoke(sa_google.refresh_token))
                if tasks:
                    anyio.run(lambda: anyio.gather(*tasks))
        except Exception:
            pass  # revoke 실패해도 로그아웃 절차는 계속

    # 3) 앱 쿠키 삭제
    clear_refresh_cookie(response)
    return {"ok": True}

# ── 디버그: 현재 redirect_uri가 어떻게 계산되는지 한 번에 보기
@app.get("/debug/redirects")
def debug_redirects(request: Request):
    return {
        "BASE_URL(env)": BASE_URL,
        "google(env)":  f"{BASE_URL}/auth/google/callback",
        "facebook(env)":f"{BASE_URL}/auth/facebook/callback",
        "naver(env)":   f"{BASE_URL}/auth/naver/callback",
        "google(runtime)":   str(request.url_for("google_callback")),
        "facebook(runtime)": str(request.url_for("facebook_callback")),
        "naver(runtime)":    str(request.url_for("naver_callback")),
    }
from fastapi import Form

# app/main.py — auth_after 전체를 아래로 교체
from fastapi import Form

@app.get("/auth/after", response_class=HTMLResponse)
def auth_after(request: Request, access_token: str, db: Session = Depends(get_db)):
    try:
        data = decode_token(access_token)
        u = db.get(User, int(data["sub"]))
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception:
        return HTMLResponse("<h3 style='color:red'>토큰이 유효하지 않습니다.</h3>", status_code=401)

    email = u.email or ""
    name  = u.name or ""

    html = f"""<!doctype html>
<html lang="ko"><meta charset="utf-8"/>
<title>계정 확인</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<body style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:560px;margin:48px auto;line-height:1.5">
  <h2>로그인 계정 확인</h2>
  <div style="padding:12px 16px;border:1px solid #ddd;border-radius:12px">
    <div><b>이름</b>: {name or "(없음)"}</div>
    <div><b>이메일</b>: {email or "(없음)"}</div>
    <small style="color:#666">ID: {u.id}</small>
  </div>

  <div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap">
    <a href="/frontend/index.html?access_token={access_token}" style="text-decoration:none;padding:10px 14px;border-radius:10px;background:#111;color:#fff;display:inline-block">이 계정으로 계속</a>
    <a href="/auth/google/login?force=true" style="text-decoration:none;padding:10px 14px;border-radius:10px;border:1px solid #333;display:inline-block">다른 Google 계정으로</a>
    <form method="post" action="/auth/logout" style="display:inline">
      <button type="submit" style="padding:10px 14px;border-radius:10px;border:1px solid #333;background:#fff">로그아웃</button>
    </form>
  </div>
</body></html>"""
    return HTMLResponse(html, status_code=200)

from pathlib import Path

VERIFY_FILE = Path(__file__).resolve().parent.parent / "google83c0ba022345b400.html"

@app.get("/google83c0ba022345b400.html", response_class=FileResponse)
def google_verify():
    if VERIFY_FILE.exists():
        return FileResponse(str(VERIFY_FILE), media_type="text/html")
    return PlainTextResponse("File not found", status_code=404)




