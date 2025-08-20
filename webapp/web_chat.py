# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_core import RAGEngine

INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b-instruct")

app = FastAPI(title="Chat RAG Web (local)", version="0.1.0")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

_engine: Optional[RAGEngine] = None

@app.on_event("startup")
def _startup():
    global _engine
    _engine = RAGEngine(INDEX_DIR, OLLAMA_URL, MODEL_NAME)

class ChatRequest(BaseModel):
    question: str
    person: Optional[str] = None
    k: int = 12
    threshold_top: float = 0.30
    threshold_mean: float = 0.28
    window_s: float = 10.0
    show_sources: bool = False

class Source(BaseModel):
    folder: str
    video_name: str
    start: float
    end: float
    speaker: Optional[str] = None
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = []

@app.get("/health")
def health():
    return {"status":"ok", "index": str(INDEX_DIR), "model": MODEL_NAME, "ollama": OLLAMA_URL}

@app.get("/api/persons")
def persons():
    return {"persons": _engine.list_persons()}

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Pregunta vacía.")
    ans, rows = _engine.answer(
        question=req.question.strip(),
        person=req.person,
        k=req.k,
        threshold_top=req.threshold_top,
        threshold_mean=req.threshold_mean,
        window_s=req.window_s
    )
    if not req.show_sources:
        return {"answer": ans, "sources": []}
    out = []
    for r in rows[:8]:
        out.append({
            "folder": r.get("folder"),
            "video_name": r.get("video_name"),
            "start": r.get("start"),
            "end": r.get("end"),
            "speaker": r.get("speaker"),
            "text": r.get("text")
        })
    return {"answer": ans, "sources": out}

@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html, status_code=200)

