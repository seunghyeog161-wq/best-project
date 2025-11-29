
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
from urllib.parse import quote
from fastapi import Query
import base64, json
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RATINGS_CSV = BASE_DIR / "ratings.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
EXCEL_PATH = BASE_DIR / "data" / "fashion_cloths_items_labeled.csv"
MEDIA_DIR = BASE_DIR / "media" / "top2"


if not EXCEL_PATH.exists():
    raise RuntimeError(f"추천용 엑셀을 찾을 수 없습니다: {EXCEL_PATH}")

# 빈칸은 NaN으로 통일
DF = pd.read_csv(EXCEL_PATH, encoding="utf-8").replace({"": np.nan, " ": np.nan})

# ===== 스타일 칼럼 자동 수집 (prob_* 로 시작하거나 괄호가 있어도 OK) =====
STYLE_COLS: List[str] = []
for c in DF.columns:
    lc = str(c).strip().lower()
    if lc.startswith("prob_"):
        STYLE_COLS.append(c)
        
        
# UI 라벨 → prob_* 컬럼 이름 매핑
STYLE_KEY_MAP = {
    "클래식":"prob_classic", "classic":"prob_classic",
    "힙합":"prob_hiphop", "hiphop":"prob_hiphop",
    "큐트":"prob_cute", "cute":"prob_cute",
    "캐주얼":"prob_casual", "casual":"prob_casual",
    "면접":"prob_interview", "interview":"prob_interview",
    "소개팅":"prob_date", "date":"prob_date",
    "소개팅(첫만남)":"prob_date(first)", "date(first)":"prob_date(first)",
    "성숙":"prob_maturity", "maturity":"prob_maturity",
    "럭셔리":"prob_luxury", "고급":"prob_luxury", "luxury":"prob_luxury",
    "영":"prob_young", "young":"prob_young",
    "섹시":"prob_sexy", "sexy":"prob_sexy",
    "여행":"prob_trip", "trip":"prob_trip",
    "아름다운":"prob_beautiful", "beautiful":"prob_beautiful",
    "러블리":"prob_lovely", "lovely":"prob_lovely",
    "유니크":"prob_unique", "unique":"prob_unique",
    "예쁜":"prob_pretty", "pretty":"prob_pretty",
    "피크닉":"prob_picnic", "picnic":"prob_picnic",
    "패턴":"prob_pattern", "pattern":"prob_pattern",
    "무서운":"prob_scary", "scary":"prob_scary",
    "인플루언서":"prob_influencer", "influencer":"prob_influencer",
    "워크":"prob_work", "work":"prob_work",
    "나이스":"prob_nice", "nice":"prob_nice",
    "파티":"prob_party", "party":"prob_party",
    "스포츠":"prob_sports", "sports":"prob_sports",
    "예술":"prob_art", "art":"prob_art",

    # ⬇⬇ 추가
    "첫 데이트":"prob_date(first)",
    "date_first":"prob_date(first)",
    "date(first)":"prob_date(first)",
}


def _find_df_style_col(target: str) -> Optional[str]:
    """target(prob_*)을 DF의 실제 컬럼명으로(case-insensitive) 매칭"""
    t = str(target).strip().lower()
    for c in STYLE_COLS:
        if str(c).strip().lower() == t:
            return c
    # 특별 처리: prob_date 가 없고 prob_date(...)만 있을 수 있음
    if t == "prob_date":
        for c in STYLE_COLS:
            if str(c).strip().lower().startswith("prob_date"):
                return c
    return None

def _resolve_style_key(k: str) -> Optional[str]:
    """라벨/별칭/생짜키 → prob_* 로 정규화 후 실제 DF 컬럼명으로 반환"""
    if not k: return None
    raw = str(k).strip().lower()
    # 1) 라벨 매핑
    mapped = STYLE_KEY_MAP.get(raw)
    if not mapped:
        # 2) 이미 prob_* 형식이면 그대로, 아니면 prob_ 접두사 붙여 시도
        mapped = raw if raw.startswith("prob_") else "prob_" + raw
    # 3) DF에 실제 존재하는 컬럼명으로 확정
    return _find_df_style_col(mapped)





# ===== 한글/영문 매핑 =====
CAT_MAP = {
    "상의": "top", "하의": "bottom", "아우터": "outer", "신발": "shoes", "액세서리": "acc",
    "top":"top","bottom":"bottom","outer":"outer","shoes":"shoes","acc":"acc",
    "accessory":"acc",
    "hat":"hat",
    "uniform":"uniform",  # ✅ 추가
}


def _cat_norm(x: str|None) -> str|None:
    if not x: return None
    s = str(x).strip().lower()
    return CAT_MAP.get(s, s)

def _gender_norm(x: str|None) -> str|None:
    if not x: return None
    s = str(x).strip().lower()
    if s in ["male","m","남","남성"]: return "male"
    if s in ["female","f","여","여성"]: return "female"
    return "both"

def _decode_q(q: str) -> dict:
    s = (q or "").strip()
    # base64 padding 보정
    s += "=" * (-len(s) % 4)
    try:
        raw = base64.b64decode(s)
    except Exception:
        # urlsafe fallback
        raw = base64.urlsafe_b64decode(s)
    return json.loads(raw.decode("utf-8"))


def _bool_from_cell(x) -> bool:
    s = str(x).strip().lower()
    return s in ["1","true","yes","y","t"]

def _media_url(rel: str|None, w=680, h=520, mode="cover") -> Optional[str]:
    if not rel: return None
    return f"/media/resize?p={quote(str(rel))}&w={w}&h={h}&mode={mode}"

# ===== 입력/출력 스키마 =====
class MatchIn(BaseModel):
    gender: Optional[str] = None
    category: Optional[str] = None              # 단일 카테고리
    categories: Optional[List[str]] = None      # ⬅️ 다중 카테고리
    styles: Optional[Dict[str, float]] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    collaborated_only: Optional[bool] = None
    uniform_only: Optional[bool] = None
    soccer_only: Optional[bool] = None
    baseball_only: Optional[bool] = None 
    basketball_only: Optional[bool] = None # ⬅️ 야구 필터
    top_k: int = Field(12, ge=1, le=50)

class ItemOut(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None
    gender: Optional[str] = None
    link: Optional[str] = None
    # ⬇️ 추가
    img_front: Optional[str] = None
    img_back: Optional[str] = None
    # 호환용(있던 필드) — front로 채워둠
    img: Optional[str] = None
    # ⬇️ 추가: 프론트가 옷번호 추출할 원본 경로
    image_path: Optional[str] = None

    style_score: float
    price_score: float
    final_score: float



class MatchOut(BaseModel):
    results: List[ItemOut]
    message: Optional[str] = None

router = APIRouter(prefix="/api/reco", tags=["recommend"])

# ===== 유틸 =====
def _norm_weights(styles: Dict[str, float] | None) -> Dict[str, float]:
    """라벨 → prob_* 변환 + DF 존재 체크 + 0~1 클리핑 (합=1 정규화 제거)"""
    if not styles:
        return {}
    tmp: Dict[str, float] = {}
    for raw_k, v in styles.items():
        col = _resolve_style_key(raw_k)
        if not col:
            continue
        try:
            x = float(v)
            if x > 1.0:  # 0~100 → 0~1
                x /= 100.0
            tmp[col] = max(0.0, min(1.0, x))
        except:
            continue
    return {str(k).strip().lower(): v for k, v in tmp.items()} 



def _user_vec(cols: List[str], w: Dict[str, float]) -> np.ndarray:
    vec = []
    for c in cols:
        key = "prob_" + str(c).strip().lower()[5:] if str(c).lower().startswith("prob_") else str(c).lower()
        vec.append(float(w.get(key, 0.0)))
    return np.array(vec, dtype=np.float32)  # ✅ 정규화 제거


def _row_vec(row: pd.Series, cols: List[str]) -> np.ndarray:
    xs = []
    for c in cols:
        try:
            x = float(row.get(c))
        except:
            x = 0.0
        if np.isnan(x): x = 0.0
        xs.append(x)
    return np.array(xs, dtype=np.float32)  # ✅ 정규화 제거


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _price_score(price: Optional[float], lo: Optional[float], hi: Optional[float]) -> float:
    if price is None or (lo is None and hi is None): return 0.5
    p = float(price)
    if lo is not None and hi is not None and hi >= lo:
        if lo <= p <= hi: return 1.0
        span = max(1.0, hi - lo)
        d = lo - p if p < lo else p - hi
        return max(0.0, 1.0 - d/span)
    if lo is not None:
        if p >= lo: return 1.0
        return max(0.0, 1.0 - (lo - p)/max(1.0, lo))
    if hi is not None:
        if p <= hi: return 1.0
        return max(0.0, 1.0 - (p - hi)/max(1.0, hi))
    return 0.5

def _apply_filters(df: pd.DataFrame, req: MatchIn) -> pd.DataFrame:
    t = df.copy()

    if "gender" in t.columns:
        t["_gender_norm"] = t["gender"].map(_gender_norm)
    if "부위" in t.columns:
        t["_cat_norm"] = t["부위"].map(_cat_norm)

    # (A) 성별: any/상관없음 이면 필터 건너뜀
    if req.gender and req.gender not in ("any", "상관없음"):
        want = _gender_norm(req.gender)
        if want in ["male", "female"]:
            t = t[t["_gender_norm"].isin([want, "both", "공용"])]

    # (B) 카테고리: 단일 or 다중
    if req.category:
        want = _cat_norm(req.category)
        t = t[t["_cat_norm"] == want]
    elif req.categories:
        wants = {_cat_norm(c) for c in req.categories if c}
        if wants:
            t = t[t["_cat_norm"].isin(wants)]

    # 콜라보/유니폼/스포츠
    if req.collaborated_only and "collaborate" in t.columns:
        t = t[t["collaborate"].map(_bool_from_cell)]

    # ✅ "부위" 컬럼이 uniform일 때만 필터링
    if req.uniform_only:
        if "uniform" in t.columns:
            t = t[
                t["uniform"].astype(str).str.lower().str.strip()
                 .isin({"1","true","y","yes","t","on","uniform"})
            ]
        elif "부위" in t.columns:
            # 혹시 '부위'에 실제로 'uniform'이 들어오는 데이터도 지원
            t = t[t["부위"].astype(str).str.lower().str.strip() == "uniform"]

    if "sport" in t.columns:
        sport_col = t["sport"].astype(str).str.lower().str.strip()
        if req.soccer_only:
            t = t[sport_col == "soccer"]
        if req.baseball_only:
            t = t[sport_col == "baseball"]
        if req.basketball_only:
            t = t[sport_col == "basketball"]

    # 가격
    if "price" in t.columns:
        t["price"] = pd.to_numeric(t["price"], errors="coerce")
        if req.min_price is not None:
            t = t[t["price"] >= float(req.min_price)]
        if req.max_price is not None:
            t = t[t["price"] <= float(req.max_price)]

    return t.reset_index(drop=True)

def provide_decode_token():
    # 순환참조 피하려고 지연 import
    from app.main import decode_token as _decode
    return _decode
@router.get("/from_q", response_model=MatchOut)
def recommend_from_q(
    q: str = Query(...),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    request: Request = None,
    decode_token = Depends(provide_decode_token),
):
    # ✅ 토큰 필수 확인
    token = _extract_token(authorization, request)
    decode_token(token)

    # q 디코드 이하 동일
    try:
        obj = _decode_q(q)
    except Exception:
        raise HTTPException(status_code=400, detail="q 파라미터 디코딩 실패")
    filters = obj.get("filters", obj)

    # 필드 꺼내기
    gender = filters.get("gender")
    cats = filters.get("categories") or []
    # styles: 배열이면 1.0 가중치로 dict 변환, dict면 그대로 사용
    styles_raw = filters.get("styles") or []
    if isinstance(styles_raw, list):
        styles = {s: 1.0 for s in styles_raw}
    elif isinstance(styles_raw, dict):
        styles = styles_raw
    else:
        styles = {}

    budget = filters.get("budget") or {}
    min_price = budget.get("min") or filters.get("min_price")
    max_price = budget.get("max") or filters.get("max_price")

    uniform_type = (filters.get("uniform_type") or filters.get("uniformType"))
    uniform_type = uniform_type.lower() if isinstance(uniform_type, str) else None

    payload = MatchIn(
        gender=gender,
        categories=cats,
        styles=styles,
        min_price=min_price,
        max_price=max_price,
        collaborated_only=filters.get("collab_only") or filters.get("collaborated_only"),
        uniform_only = "uniform" in [c.lower() for c in cats],
        soccer_only=(uniform_type == "soccer"),
        baseball_only=(uniform_type == "baseball"), 
        basketball_only=(uniform_type == "basketball"),
        top_k=int(filters.get("top_k") or 12),
    )
    return _compute_recommendation(payload)

# --- REPLACE: robust token extractor with logs ---
def _safe(tok: str) -> str:
    return (tok or "")[:12] + "..." if tok else "-"

def _extract_token(authorization: Optional[str], request: Request) -> str:
    # 1) Authorization 헤더 우선
    if authorization:
        tok = authorization.strip()
        if tok.lower().startswith("bearer "):
            tok = tok.split(None, 1)[1]
        if tok:
            print("[auth] via header:", _safe(tok))
            return tok



    # 3) 쿠키 (여러 키 지원, 'Bearer ' 접두사 제거)
    cookie_keys = ("access_token","accessToken","jwt","token","id_token","authToken","authorization")
    for k in cookie_keys:
        v = request.cookies.get(k)
        if v:
            v = v.strip()
            if v.lower().startswith("bearer "):
                v = v.split(None, 1)[1]
            if v:
                print(f"[auth] via cookie:{k} ->", _safe(v))
                return v

    print("[auth] no token found in header/query/cookie")
    raise HTTPException(status_code=401, detail="Missing access token")
# 파일 상단 util 근처에 추가
def _pick_local_images(rel: str | None, w=680, h=520, mode="cover") -> tuple[Optional[str], Optional[str]]:
    """
    ① CSV가 "586_infront.jpg,586_back.jpg" 처럼 콤마 2개 파일이면 그대로 매핑
    ② 파일명만 와도 MEDIA_DIR 아래에서 찾음(glob/rglob)
    ③ 아니면 기존(단일 파일/폴더) 추정 로직 수행
    """
    if not rel:
        return (None, None)

    def _resize_url_by_rel(rel_file: Path) -> str:
        rel_posix = rel_file.as_posix()  # 백슬래시 -> 슬래시
        return f"/media/resize?p={quote(rel_posix)}&w={w}&h={h}&mode={mode}"

    # 0) 콤마 분리 우선 처리
    parts = [p.strip() for p in str(rel).split(",") if p.strip()]
    if parts:
        def _locate(name: str) -> Optional[str]:
            # 0-1) MEDIA_DIR/name
            p = (MEDIA_DIR / name).resolve()
            if p.exists():
                return _resize_url_by_rel(p.relative_to(MEDIA_DIR))
            # 0-2) 루트/하위 전체에서 파일명 검색
            cand = next(MEDIA_DIR.glob(name), None) or next(MEDIA_DIR.rglob(name), None)
            if cand:
                return _resize_url_by_rel(cand.relative_to(MEDIA_DIR))
            return None

        front = _locate(parts[0]) if len(parts) >= 1 else None
        back  = _locate(parts[1]) if len(parts) >= 2 else None
        if front or back:
            return (front, back)

    # 1) 기존: 단일 파일/폴더 추정
    rel_path = Path(str(rel))
    base = (MEDIA_DIR / rel_path).resolve()
    try:
        if not str(base).lower().startswith(str(MEDIA_DIR.resolve()).lower()):
            return (None, None)
    except Exception:
        return (None, None)

    def _resize_url(p: Path) -> str:
        rel_file = p.relative_to(MEDIA_DIR)
        return _resize_url_by_rel(rel_file)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    def normalize(s: str) -> str:
        return s.strip().lower()

    if base.is_file():
        parent = base.parent
        files = [p for p in parent.iterdir() if p.is_file() and p.suffix.lower() in exts]
        name_l = normalize(base.name)
        stem = normalize(base.stem)
        prefix = re.split(r'[_\-. ]+', stem)[0]
        front = back = None

        if any(x in name_l for x in ["infront", "infron", "front"]):
            front = _resize_url(base)
            for p in files:
                if prefix in normalize(p.stem) and "back" in normalize(p.name):
                    back = _resize_url(p); break

        elif "back" in name_l:
            back = _resize_url(base)
            for p in files:
                if prefix in normalize(p.stem) and any(x in normalize(p.name) for x in ["infront","infron","front"]):
                    front = _resize_url(p); break
        else:
            front = _resize_url(base)
            for p in files:
                if p == base: continue
                if prefix in normalize(p.stem) and "back" in normalize(p.name):
                    back = _resize_url(p); break

        return (front, back)

    if base.is_dir():
        front = back = None
        files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts]
        for p in files:
            name = normalize(p.name)
            if "infront" in name or "infron" in name:
                front = _resize_url(p); break
        if not front:
            for p in files:
                if "front" in normalize(p.name):
                    front = _resize_url(p); break
        if not front and files:
            front = _resize_url(files[0])

        for p in files:
            if "back" in normalize(p.name):
                back = _resize_url(p); break
        if not back and len(files) > 1:
            back = _resize_url(files[1])

        return (front, back)

    return (None, None)




# ===== 메인 엔드포인트 =====
@router.post("/best", response_model=MatchOut)
def recommend_best(
    payload: MatchIn,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    request: Request = None,
    decode_token = Depends(provide_decode_token),
):
    # ✅ 헤더/쿠키 어디서든 토큰 추출 + 검증
    token = _extract_token(authorization, request)
    try:
        decode_token(token)
    except HTTPException as e:
        # decode_token 이 자체적으로 HTTPException을 던지는 경우 그대로 전달
        raise
    except Exception as e:
        # 만료/서명오류 등 공통 처리 (라이브러리별 예외명을 한 줄로 표시)
        raise HTTPException(status_code=401, detail=f"Invalid token ({type(e).__name__})")

    return _compute_recommendation(payload)

 
def _compute_recommendation(payload: MatchIn) -> MatchOut:
    if len(STYLE_COLS) == 0:
        raise HTTPException(status_code=500, detail="엑셀에 prob_* 스타일 컬럼이 없습니다.")

    df = _apply_filters(DF, payload)
    if df.empty:
        return MatchOut(results=[], message="조건에 맞는 후보가 없습니다.")

    W = _norm_weights(payload.styles)
    u = _user_vec(STYLE_COLS, W)

    style_scores = []
    for _, row in df.iterrows():
        v = _row_vec(row, STYLE_COLS)
        score = float(np.dot(u, v)) if np.any(u) else 0.0
        style_scores.append(score)

    style_scores = np.array(style_scores, dtype=np.float32)
    final = style_scores  # 가격은 이미 엄격 필터로 제외됨

    df = (
        df.assign(_style=style_scores, _price=1.0, _final=final, _tie=np.random.rand(len(df)))
          .sort_values(["_final","_tie"], ascending=[False, False])
    )
    # ✅ 요청에서 온 top_k 사용
    k = max(1, min(int(getattr(payload, "top_k", 12) or 12), 50))
    top = df.head(k)

    outs: List[ItemOut] = []
    for _, r in top.iterrows():
        img_front, img_back = _pick_local_images(r.get("image_path"), w=960, h=720)
        outs.append(ItemOut(
            name        = r.get("name"),
            brand       = r.get("브랜드"),
            price       = float(r.get("price")) if pd.notna(r.get("price")) else None,
            category    = r.get("부위"),
            gender      = r.get("gender"),
            link        = r.get("product_url"),
            img_front   = img_front or None,   # ✅ 콤마 필수
            img_back    = img_back  or None,   # ✅ 콤마 필수
            img         = img_front,           # ✅ 쉼표 제거 (튜플 방지)
            image_path  = r.get("image_path"), # ✅ 프론트로 전달
            style_score = float(r.get("_style")),
            price_score = float(r.get("_price")),
            final_score = float(r.get("_final")),
        ))

    return MatchOut(results=outs, message="가격은 필터로 엄격 적용, 스타일 유사도만으로 상위 추천을 반환합니다.")



import csv
from datetime import datetime
from pathlib import Path

@router.post("/feedback/rating")
def save_rating(
    item: dict,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    request: Request = None,
    decode_token = Depends(provide_decode_token),
):
    # 인증
    token = _extract_token(authorization, request)
    decode_token(token)

    # 기본 필드
    item_name  = item.get("item_name")
    link       = item.get("link")
    image_path = item.get("image_path")
    rating_raw = item.get("rating")

    if not item_name or rating_raw is None:
        raise HTTPException(status_code=400, detail="item_name 또는 rating 누락")

    try:
        rating = int(rating_raw)
    except:
        raise HTTPException(status_code=400, detail="rating은 1~5 정수여야 합니다.")
    if not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="rating은 1~5 범위여야 합니다.")

    # 파일명만 추출
    img_filename = "-"
    if image_path:
        try:
            img_filename = Path(image_path).stem
        except:
            pass

    # ★ 추천 컨텍스트
    rec_gender      = item.get("rec_gender")      or "-"
    rec_category    = item.get("rec_category")    or "-"
    rec_categories  = item.get("rec_categories")  or []   # list 가능
    rec_styles      = item.get("rec_styles")      or {}   # dict 가능
    rec_styles_str  = item.get("rec_styles_str")  or "-"  # "힙합0.99,클래식0.66"
    rec_min_price   = item.get("rec_min_price")
    rec_max_price   = item.get("rec_max_price")

    # 문자열화
    cats_str    = "|".join(map(str, rec_categories)) if isinstance(rec_categories, list) and rec_categories else (str(rec_categories) if rec_categories else "-")
    styles_json = json.dumps(rec_styles, ensure_ascii=False) if isinstance(rec_styles, dict) and rec_styles else "-"
    min_str = "-" if rec_min_price in (None, "") else str(rec_min_price)
    max_str = "-" if rec_max_price in (None, "") else str(rec_max_price)

    # CSV 저장
    try:
        RATINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(RATINGS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 열 순서: 기존 5개 + 컨텍스트 7개
            writer.writerow([
                datetime.now().isoformat(),  # timestamp
                item_name,                   # item_name
                link or "-",                 # link
                img_filename,                # img_filename
                rating,                      # rating
                rec_gender,                  # ★ 남/여/상관없음
                rec_category,                # ★ 단일 카테고리
                cats_str,                    # ★ 다중 카테고리 "상의|아우터"
                styles_json,                 # ★ 스타일 JSON {"힙합":0.99,...}
                rec_styles_str,              # ★ 요약 "힙합0.99,클래식0.66"
                min_str,                     # ★ 희망 최저가
                max_str,                     # ★ 희망 최고가
            ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 저장 실패: {str(e)}")

    return {"ok": True, "message": "별점 + 추천조건 저장 완료"}

