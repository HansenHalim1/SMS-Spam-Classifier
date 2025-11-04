import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
ART_DIR = (APP_DIR.parent / "artifacts").resolve()

DEFAULT_ORIGINS = [
    "http://localhost:8501",
    "http://localhost:3000",
    "https://*.hf.space",
]


def parse_origins() -> List[str]:
    env = os.getenv("ALLOWED_ORIGINS", "")
    if not env.strip():
        return DEFAULT_ORIGINS
    return [origin.strip() for origin in env.split(",") if origin.strip()]


with open(ART_DIR / "feature_config.json", "r", encoding="utf-8") as fh:
    FEATURE_CFG = json.load(fh)
with open(ART_DIR / "label_map.json", "r", encoding="utf-8") as fh:
    LABEL_MAP = {int(key): value for key, value in json.load(fh).items()}
with open(ART_DIR / "model.pkl", "r", encoding="utf-8") as fh:
    MODEL = json.load(fh)

MODEL_VERSION = MODEL.get("version", "v0")

VOCAB: Dict[str, int] = {token: int(idx) for token, idx in MODEL["vocabulary"].items()}
IDF = np.array(MODEL["idf"], dtype=np.float32).reshape(1, -1)
COEF = np.array(MODEL["coef"], dtype=np.float32).reshape(1, -1)
BIAS = float(MODEL["intercept"])

TOK_RE = re.compile(FEATURE_CFG.get("token_pattern", r"[a-zA-Z']+"))


def tokenize(text: str) -> List[str]:
    if FEATURE_CFG.get("lowercase", True):
        text = text.lower()
    return TOK_RE.findall(text or "")


def tfidf_vector(tokens: List[str]) -> np.ndarray:
    vector = np.zeros((1, len(VOCAB)), dtype=np.float32)
    if not tokens:
        return vector

    for token in tokens:
        idx = VOCAB.get(token)
        if idx is not None:
            vector[0, idx] += 1.0

    if vector.sum() == 0:
        return vector

    vector = 1.0 + np.log(vector, out=np.zeros_like(vector), where=(vector > 0))
    vector = vector * IDF

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def predict_proba(text: str):
    tokens = tokenize(text)
    vector = tfidf_vector(tokens)
    logit = float(vector @ COEF.T + BIAS)
    prob_spam = sigmoid(logit)
    return prob_spam, tokens


class PredictIn(BaseModel):
    text: str


class PredictOut(BaseModel):
    prediction: str
    score: float
    latency_ms: int
    model_version: str
    debug: Dict[str, Any]


app = FastAPI(title="SMS Spam Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    if not isinstance(payload.text, str) or not payload.text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' must be a non-empty string.")

    start = time.time()
    prob_spam, tokens = predict_proba(payload.text)
    label_idx = 1 if prob_spam >= 0.5 else 0
    label = LABEL_MAP[label_idx]
    latency = int((time.time() - start) * 1000)

    contributions: List[Tuple[str, float]] = []
    for token in set(tokens):
        if token not in VOCAB:
            continue
        component = tfidf_vector([token])
        score = float(component @ COEF.T)
        contributions.append((token, round(score, 3)))
    contributions.sort(key=lambda item: item[1], reverse=True)
    contributions = contributions[:10]

    score = prob_spam if label_idx == 1 else 1 - prob_spam

    return {
        "prediction": label,
        "score": round(float(score), 4),
        "latency_ms": latency,
        "model_version": MODEL_VERSION,
        "debug": {"top_features": contributions},
    }
