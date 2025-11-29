# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

DATA_DIR = Path("./data")
ITEMS_CSV = DATA_DIR / "items.csv"   # 네가 가진 라벨 결과를 CSV로 저장해 두세요

# ===== 데이터 로드 =====
if not ITEMS_CSV.exists():
    raise RuntimeError(f"items.csv가 필요합니다: {ITEMS_CSV.resolve()}")
ITEMS = pd.read_csv(ITEMS_CSV)

# 사용할 prob_* 컬럼 자동 수집
PROB_COLS = [c for c in ITEMS.columns if str(c).startswith("prob_")]
CATEGORY_MAP = {"상의": "top", "아우터": "outer", "top": "top", "outer": "outer"}

def _norm_weights(d: Dict[str, float]) -> Dict[str, float]:
    # 0~100 -> 0~1 -> 합 1
    sw = {k: (float(v)/100.0 if float(v) > 1.0 else float(v)) for k, v in d.items()}
    s = sum(sw.values())
    if s <= 0:
        return {k: 1.0/len(sw) for k in sw} if sw else {}
    return {k: v/s for k, v in sw.items()}

def _style_score(df: pd.DataFrame, sw: Optional[Dict[str, float]]) -> np.ndarray:
    """사용자 가중치와 prob_* 내적한 점수"""
    if not sw: 
        return np.zeros((len(df),), dtype=np.float32)
    use = {}
    for k, w in sw.items():
        col = f"prob_{k}"
        if col in df.columns:
            use[col] = float(w)
    if not use:
        return np.zeros((len(df),), dtype=np.float32)
    acc = np.zeros((len(df),), dtype=np.float32)
    for col, w in use.items():
        acc += w * df[col].fillna(0).astype(float).to_numpy(dtype=np.float32)
    return acc

def recommend_once(
    category: str,                      # "상의" | "아우터"
    gender: Optional[str] = None,       # "male" | "female"
    uniform: bool = False,
    sport: Optional[str] = None,        # "soccer" | "baseball"
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    style_weights: Optional[Dict[str, float]] = None,  # {"classic":80, "hiphop":20}
):
    if category not in CATEGORY_MAP:
        raise ValueError("category는 상의/아우터 중 하나여야 합니다.")

    df = ITEMS.copy()

    # 1) 카테고리(부위) 필터
    if "부위" not in df.columns:
        raise ValueError("items.csv에 '부위' 컬럼이 필요합니다.")
    want = "상의" if CATEGORY_MAP[category] == "top" else "아우터"
    df = df[df["부위"].astype(str).str.strip() == want]

    # 2) 성별(있으면)
    # 2) 성별(있으면)
    if gender and "gender" in df.columns:
        g = gender.strip().lower()
        if g in ("male", "남성"):
            df = df[df["gender"].astype(str).str.lower() == "male"]
        elif g in ("female", "여성"):
            df = df[df["gender"].astype(str).str.lower() == "female"]
        elif g in ("both", "상관없음"):
            df = df[df["gender"].astype(str).str.lower() == "both"]


    # 3) 유니폼/스포츠
    # 3) 유니폼 및 스포츠 필터링
    if uniform:
        df = df[df["uniform"] == 1]  # 유니폼 선택 시 필터링

        # 스포츠 선택이 있는 경우
        if sport:
            sp = sport.strip().lower()
            
            # sport 컬럼이 있다면 우선 사용
            if "sport" in df.columns:
                df = df[df["sport"].astype(str).str.lower() == sp]
            else:
                # sport 컬럼이 없고, 종목별 컬럼이 따로 있을 경우
                if sp == "soccer" and "soccer" in df.columns:
                    df = df[df["soccer"] == 1]
                elif sp == "baseball" and "baseball" in df.columns:
                    df = df[df["baseball"] == 1]
    else:
        df = df[df["uniform"] == 0]  # 유니폼 선택 안 했으면 일반 의류 필터링


    # 4) 가격
    if price_min is not None and "price" in df.columns:
        df = df[df["price"].astype(float) >= float(price_min)]
    if price_max is not None and "price" in df.columns:
        df = df[df["price"].astype(float) <= float(price_max)]

    if len(df) == 0:
        raise ValueError("조건에 맞는 아이템이 없습니다.")

    # 5) 스타일 가중치 점수
    sw = _norm_weights(style_weights) if style_weights else None
    s = _style_score(df, sw)
    best = int(np.argmax(s))
    row = df.iloc[best]

    need = {"name", "product_url", "price"}
    if not need.issubset(row.index):
        raise ValueError("items.csv에 name/product_url/price 컬럼이 필요합니다.")

    return {
        "name": str(row["name"]),
        "link": str(row["product_url"]),
        "price": int(float(row["price"])),
        "score": float(s[best]),
    }

# ============ FastAPI =============
class RecoIn(BaseModel):
    category: str = Field(..., description="상의|아우터")
    gender: Optional[str] = None
    uniform: bool = False
    sport: Optional[str] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    style_weights: Optional[Dict[str, float]] = None  # ex) {"classic":80,"hiphop":20}

app = FastAPI(title="추천 최소 API")

@app.post("/recommend")
def recommend_api(req: RecoIn):
    try:
        return recommend_once(
            category=req.category, gender=req.gender, uniform=req.uniform, sport=req.sport,
            price_min=req.price_min, price_max=req.price_max, style_weights=req.style_weights,
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

if __name__ == "__main__":
    # 서버로 띄우기:  python reco_min.py
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False)
