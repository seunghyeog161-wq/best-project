from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
import httpx

from data.config import settings
from data.db import get_db
from data.models import User
from authlib.integrations.starlette_client import OAuth

router = APIRouter(prefix="/auth", tags=["auth"])

# ─────────────────────────────────────────────────────────────────
# OAuth 등록
# ─────────────────────────────────────────────────────────────────
oauth = OAuth()

oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
    redirect_uri=f"{settings.BASE_URL}/auth/google/callback",
)

oauth.register(
    name="naver",
    client_id=settings.NAVER_CLIENT_ID,
    client_secret=settings.NAVER_CLIENT_SECRET,
    authorize_url="https://nid.naver.com/oauth2.0/authorize",
    access_token_url="https://nid.naver.com/oauth2.0/token",
    api_base_url="https://openapi.naver.com/v1/nid/",
    client_kwargs={"scope": "name email profile_image"},
    redirect_uri=f"{settings.BASE_URL}/auth/naver/callback",
)

FRONTEND_ORIGIN = settings.FRONTEND_ORIGIN

# ─────────────────────────────────────────────────────────────────
# JWT
# ─────────────────────────────────────────────────────────────────
def issue_jwt(subject: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)

def set_auth_cookies(resp: RedirectResponse, jwt_token: str, user: User):
    # 개발(HTTP): Lax/secure False. 배포(HTTPS): None/True/도메인 지정 권장
    resp.set_cookie(
        "novic_access_token", jwt_token,
        httponly=True, samesite="Lax", secure=False,
        max_age=60*60*24*30, path="/"
    )
    resp.set_cookie(
        "novic_name", user.name or "",
        httponly=False, max_age=60*60*24*7, path="/"
    )
    resp.set_cookie(
        "novic_avatar", user.avatar or "",
        httponly=False, max_age=60*60*24*7, path="/"
    )

# ─────────────────────────────────────────────────────────────────
# 로그인 시작
# ─────────────────────────────────────────────────────────────────
@router.get("/google/login")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request)

@router.get("/naver/login")
async def naver_login(request: Request):
    return await oauth.naver.authorize_redirect(request)

# ─────────────────────────────────────────────────────────────────
# 콜백: Google
# ─────────────────────────────────────────────────────────────────
@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    if "google" not in oauth._clients:
        raise HTTPException(500, "Google OAuth is not configured")

    token = await oauth.google.authorize_access_token(request)

    # 1) id_token 파싱(가장 신뢰도 높음) → 2) userinfo 엔드포인트 백업
    userinfo = None
    try:
        userinfo = await oauth.google.parse_id_token(request, token)
    except Exception:
        pass

    if not userinfo:
        resp = await oauth.google.get("userinfo", token=token)
        userinfo = resp.json()

    gid = userinfo.get("sub") or userinfo.get("id")
    email = userinfo.get("email")
    name = userinfo.get("name")
    avatar = userinfo.get("picture")

    if not gid:
        raise HTTPException(400, "No Google user id")

    # upsert
    user = db.query(User).filter_by(provider="google", provider_id=str(gid)).first()
    if not user:
        # 같은 이메일 합칠 정책 원하면 다음 라인 주석 해제:
        # if email: user = db.query(User).filter_by(email=email).first()
        if not user:
            user = User(provider="google", provider_id=str(gid))
            db.add(user)

    user.email = email or user.email
    user.name = name or user.name
    user.avatar = avatar or user.avatar
    db.commit()
    db.refresh(user)

    jwt_token = issue_jwt(email or str(gid))
    resp = RedirectResponse(url="/auth/success", status_code=302)
    set_auth_cookies(resp, jwt_token, user)
    return resp

# ─────────────────────────────────────────────────────────────────
# 콜백: Naver
# ─────────────────────────────────────────────────────────────────
@router.get("/naver/callback")
async def naver_callback(request: Request, db: Session = Depends(get_db)):
    if "naver" not in oauth._clients:
        raise HTTPException(500, "Naver OAuth is not configured")

    token = await oauth.naver.authorize_access_token(request)

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {token['access_token']}"}
        )
    res = r.json() or {}
    profile = res.get("response", {})

    nid = profile.get("id")
    email = profile.get("email")
    name = profile.get("name") or profile.get("nickname")
    avatar = profile.get("profile_image")

    if not nid:
        raise HTTPException(400, "No Naver user id")

    # upsert
    user = db.query(User).filter_by(provider="naver", provider_id=str(nid)).first()
    if not user:
        # 이메일 병합 원하면: if email: user = db.query(User).filter_by(email=email).first()
        if not user:
            user = User(provider="naver", provider_id=str(nid))
            db.add(user)

    user.email = email or user.email
    user.name = name or user.name
    user.avatar = avatar or user.avatar
    db.commit()
    db.refresh(user)

    jwt_token = issue_jwt(email or str(nid))
    resp = RedirectResponse(url="/auth/success", status_code=302)
    set_auth_cookies(resp, jwt_token, user)
    return resp

# ─────────────────────────────────────────────────────────────────
# 성공 페이지(프론트로 통일 리다이렉트)
# ─────────────────────────────────────────────────────────────────
@router.get("/success", response_class=HTMLResponse)
async def auth_success():
    # SPA는 쿠키만 있으면 로그인 상태 판단 가능
    return HTMLResponse(f"""
    <script>
      window.location.href = "{FRONTEND_ORIGIN}/";
    </script>
    """)

# ─────────────────────────────────────────────────────────────────
# 유틸: 로그아웃 & 상태 확인(개발용)
# ─────────────────────────────────────────────────────────────────
@router.get("/logout")
async def logout():
    resp = RedirectResponse(url=f"{FRONTEND_ORIGIN}/", status_code=302)
    resp.delete_cookie("novic_access_token", path="/")
    resp.delete_cookie("novic_name", path="/")
    resp.delete_cookie("novic_avatar", path="/")
    return resp

@router.get("/me")
async def me(request: Request):
    token = request.cookies.get("novic_access_token")
    if not token:
        return {"authenticated": False}
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        return {"authenticated": True, "sub": payload.get("sub")}
    except JWTError:
        return {"authenticated": False}
