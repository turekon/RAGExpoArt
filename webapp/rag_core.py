# -*- coding: utf-8 -*-
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# -------------------- utilidades E/S --------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

# -------------------- encoder --------------------
def load_encoder(model_name: str):
    return SentenceTransformer(model_name, trust_remote_code=True)

def embed_texts(encoder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Embebe textos garantizando list[str] y normalizando embeddings."""
    vecs: List[np.ndarray] = []
    dim = getattr(encoder, "get_sentence_embedding_dimension", lambda: 768)()
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        clean: List[str] = []
        for t in batch:
            if t is None:
                clean.append("")
            elif isinstance(t, str):
                clean.append(t)
            elif isinstance(t, bytes):
                clean.append(t.decode("utf-8", errors="ignore"))
            else:
                clean.append(str(t))
        if not clean:
            continue
        part = encoder.encode(
            clean,
            batch_size=min(batch_size, len(clean)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        vecs.append(part)
    if not vecs:
        return np.zeros((0, int(dim)), dtype="float32")
    return np.vstack(vecs)

# -------------------- índice --------------------
def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta = read_jsonl(index_dir / "meta.jsonl")
    enc_info = json.loads((index_dir / "encoder.json").read_text(encoding="utf-8"))
    return index, meta, enc_info

# -------------------- búsqueda --------------------
def mmr_from_index(index, query_vec: np.ndarray, candidate_ids: List[int], topn: int, lambda_: float = 0.5):
    """MMR reconstruyendo solo vectores de candidatos (IndexFlatIP soporta reconstruct)."""
    cand_vecs = []
    for cid in candidate_ids:
        v = np.asarray(index.reconstruct(int(cid)), dtype="float32")
        cand_vecs.append(v)
    if not cand_vecs:
        return [], {}
    cand_vecs = np.vstack(cand_vecs)
    sims = (cand_vecs @ query_vec.reshape(-1,1)).ravel()
    order = sims.argsort()[::-1].tolist()
    selected = []
    while order and len(selected) < topn:
        if not selected:
            selected.append(order.pop(0)); continue
        best_j, best_score = None, -1e9
        for j in order:
            sim_q = float(sims[j])
            sim_sel = max(float(cand_vecs[j] @ cand_vecs[s]) for s in selected)
            score = lambda_ * sim_q - (1.0 - lambda_) * sim_sel
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j); order.remove(best_j)
    picked_ids = [int(candidate_ids[j]) for j in selected]
    sim_map = {int(candidate_ids[j]): float(sims[j]) for j in range(len(candidate_ids))}
    return picked_ids, sim_map

def seconds_to_hhmmss(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600); m = int((x % 3600) // 60); s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_context(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for r in rows:
        t0 = seconds_to_hhmmss(r["start"]); t1 = seconds_to_hhmmss(r["end"])
        who = r.get("speaker") or "DESCONOCIDO"
        src = f"{r['folder']}/{r['video_name']} {t0}-{t1}"
        text = (r.get("text") or "").replace("\n", " ").strip()
        lines.append(f"[{who}] ({src}) {text}")
    return "\n".join(lines)

def expand_context(meta, seed_rows, window_s: float, max_total: int = 60):
    selected = {(r["video_relpath"], r["start"], r["end"]) for r in seed_rows}
    out = list(seed_rows)
    for r in seed_rows:
        vrel, a, b = r["video_relpath"], float(r["start"]), float(r["end"])
        lo, hi = a - window_s, b + window_s
        for m in meta:
            if m["video_relpath"] != vrel:
                continue
            s, e = float(m["start"]), float(m["end"])
            key = (vrel, s, e)
            if key in selected:
                continue
            if (s >= lo and s <= hi) or (e >= lo and e <= hi):
                out.append(m); selected.add(key)
                if len(out) >= max_total:
                    return sorted(out, key=lambda x: (x["folder"], x["video_name"], x["start"]))
    return sorted(out, key=lambda x: (x["folder"], x["video_name"], x["start"]))

def filter_candidates_by_person(candidates_ids, meta, person: Optional[str]):
    if not person:
        return candidates_ids
    pl = person.lower()
    return [i for i in candidates_ids if (meta[i].get("folder") or "").lower() == pl]

# -------------------- prompts y LLM --------------------
def make_system_prompt(locale: str) -> str:
    return (
        "Eres un asistente que responde ÚNICAMENTE con base en el CONTEXTO proporcionado. "
        "Integra y sintetiza ideas de múltiples fragmentos. "
        "Si el contexto no contiene la respuesta, escribe exactamente: 'No sé'. "
        "No inventes datos. Responde en español (Colombia)."
    )

def make_user_prompt(question: str, context: str) -> str:
    return (
        f"Pregunta: {question}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        "Instrucción: Responde con la mejor síntesis posible usando solo el CONTEXTO. "
        "Si el contexto no contiene la respuesta, escribe exactamente 'No sé'."
    )

def ollama_chat_generate(model: str, system_prompt: str, user_prompt: str, base_url: str = "http://localhost:11434") -> str:
    url = f"{base_url}/api/generate"
    prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 512},
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

# -------------------- Engine --------------------
class RAGEngine:
    def __init__(self, index_dir: Path, ollama_url: str, model_name: str, embed_model_override: Optional[str] = None):
        self.index, self.meta, self.enc_info = load_index(index_dir)
        self.ollama_url = ollama_url
        self.model_name = model_name
        embed_model = embed_model_override or self.enc_info["model"]
        self.encoder = load_encoder(embed_model)
        self.is_e5 = "e5" in (embed_model or "").lower()
        self.persons = sorted({ (m.get("folder") or "").strip() for m in self.meta if m.get("folder") })
    
    def list_persons(self) -> List[str]:
        return self.persons

    def answer(self, question: str, person: Optional[str], k: int, threshold_top: float, threshold_mean: float, window_s: float) -> Tuple[str, List[Dict[str,Any]]]:
        q_input = question
        if self.is_e5 and not question.lower().startswith("query:"):
            q_input = f"query: {question}"
        q_vec = embed_texts(self.encoder, [q_input])
        if q_vec.shape[0] == 0:
            return "No sé", []

        fetch_k = min(k * 6, 150)
        D, I = self.index.search(q_vec, fetch_k)
        candidates = [int(i) for i in I[0].tolist() if i != -1]
        candidates = filter_candidates_by_person(candidates, self.meta, person)
        if not candidates:
            return "No sé", []

        picked, sims_map = mmr_from_index(self.index, q_vec[0], candidates, topn=k, lambda_=0.5)

        scores = sorted([float(sims_map[i]) for i in picked], reverse=True)
        top1 = scores[0] if scores else 0.0
        top3 = float(np.mean(scores[:3])) if scores else 0.0
        if (top1 < threshold_top) or (top3 < threshold_mean):
            return "No sé", []

        seed = [self.meta[i] for i in picked]
        ctx_rows = expand_context(self.meta, seed, window_s=window_s, max_total=max(k*5, 60))
        context = format_context(ctx_rows)
        sys_p = make_system_prompt("es-CO")
        usr_p = make_user_prompt(question, context)
        ans = ollama_chat_generate(self.model_name, sys_p, usr_p, base_url=self.ollama_url)
        return ans, ctx_rows

