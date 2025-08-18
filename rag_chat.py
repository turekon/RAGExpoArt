#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, sys, time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import requests
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# ---------- Embeddings ----------
def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    # normalize_embeddings=True facilita usar FAISS IP como coseno
    return SentenceTransformer(model_name, trust_remote_code=True)

def embed_texts(encoder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        part = encoder.encode(
            texts[i:i+batch_size],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        vecs.append(part.astype("float32"))
    return np.vstack(vecs) if vecs else np.zeros((0, 384), dtype="float32")

# ---------- IO JSONL ----------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Indexado ----------
def build_index(corpus_path: Path, out_dir: Path, embed_model: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(corpus_path)
    if not data:
        console.print("[red]Corpus vacío o ilegible[/red]")
        sys.exit(1)

    # Texto base por segmento (incluye rol si existe)
    texts = []
    metas = []
    for i, r in enumerate(data):
        spk = r.get("speaker")
        seg_txt = (r.get("text") or "").strip()
        base = f"[{spk}] {seg_txt}" if spk else seg_txt
        # Puedes añadir más señales: carpeta, nombre de video, etc.
        texts.append(base)
        metas.append({
            "id": i,
            "video_relpath": r.get("video_relpath"),
            "video_name": r.get("video_name"),
            "folder": r.get("folder"),
            "start": r.get("start"),
            "end": r.get("end"),
            "speaker": spk,
            "text": seg_txt
        })

    encoder = load_encoder(embed_model)
    vecs = embed_texts(encoder, texts)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # IP con vectores normalizados = coseno
    index.add(vecs)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    write_jsonl(out_dir / "meta.jsonl", metas)
    (out_dir / "encoder.json").write_text(json.dumps({
        "model": embed_model,
        "dim": int(d),
        "count": len(metas)
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(f"[green]OK[/green] Index guardado en: {out_dir}")

def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, topn: int, fetch_k: int = 50, lambda_: float = 0.5):
    # Devuelve índices MMR (diversidad) a partir de similitudes
    sims = (doc_vecs @ query_vec.reshape(-1,1)).ravel()
    candidates = sims.argsort()[::-1][:fetch_k].tolist()
    selected = []
    while candidates and len(selected) < topn:
        if not selected:
            selected.append(candidates.pop(0))
            continue
        # Penalizar por similitud con lo ya elegido
        mmr_scores = []
        for c in candidates:
            sim_to_query = sims[c]
            sim_to_selected = max((doc_vecs[c] @ doc_vecs[s]) for s in selected)
            score = lambda_ * sim_to_query - (1 - lambda_) * sim_to_selected
            mmr_scores.append((score, c))
        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        selected.append(mmr_scores[0][1])
        candidates.remove(mmr_scores[0][1])
    return selected, sims

def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta = read_jsonl(index_dir / "meta.jsonl")
    enc_info = json.loads((index_dir / "encoder.json").read_text(encoding="utf-8"))
    return index, meta, enc_info

# ---------- LLM (Ollama) ----------
def ollama_chat(model: str, system_prompt: str, user_prompt: str, base_url: str = "http://localhost:11434"):
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096
        },
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "").strip()

def seconds_to_hhmmss(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600); m = int((x % 3600) // 60); s = int(x % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_context(rows: List[Dict[str, Any]]) -> str:
    # Breve, citable por el LLM
    lines = []
    for r in rows:
        t0 = seconds_to_hhmmss(r["start"])
        t1 = seconds_to_hhmmss(r["end"])
        who = r["speaker"] or "DESCONOCIDO"
        src = f"{r['folder']}/{r['video_name']} {t0}-{t1}"
        text = r["text"].replace("\n", " ").strip()
        lines.append(f"[{who}] ({src}) {text}")
    return "\n".join(lines)

def make_system_prompt(locale: str) -> str:
    return (
        "Eres un asistente que responde únicamente con base en el CONTEXTO dado. "
        "Si la pregunta no está cubierta por el contexto, responde exactamente: 'No sé'. "
        "No inventes información. Responde en español, registro neutro de Colombia."
    )

def make_user_prompt(question: str, context: str) -> str:
    return (
        f"Pregunta: {question}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        "Instrucción: Responde con la mejor síntesis posible usando solo el CONTEXTO. "
        "Si el contexto no contiene la respuesta, escribe exactamente 'No sé'."
    )

# ---------- CLI ----------
def cmd_index(args):
    build_index(Path(args.corpus), Path(args.out), args.embed_model)
    if args.show_stats:
        _, meta, info = load_index(Path(args.out))
        table = Table(title="Índice")
        table.add_column("count"); table.add_column("dim"); table.add_column("encoder")
        table.add_row(str(info["count"]), str(info["dim"]), info["model"])
        console.print(table)

def cmd_chat(args):
    index, meta, enc_info = load_index(Path(args.index))
    encoder = load_encoder(enc_info["model"])

    console.print("[bold]Chat RAG local[/bold] — escribe tu pregunta (Ctrl+C para salir)")
    while True:
        try:
            q = console.input("[cyan]Tú[/cyan]: ").strip()
            if not q:
                continue

            q_vec = embed_texts(encoder, [q])
            # búsqueda bruta (IP); FAISS retorna índices y distancias
            D, I = index.search(q_vec, min(args.k * 3, 50))
            # MMR para diversidad
            idx_mmr, sims = mmr(q_vec[0], index.reconstruct_n(0, index.ntotal), args.k, fetch_k=min(args.k*6, 100), lambda_=0.5)
            # combinar top por similitud y MMR (simple unión priorizando MMR)
            picked = []
            for i in idx_mmr:
                if i not in picked:
                    picked.append(i)
                if len(picked) >= args.k:
                    break

            top_scores = sorted([float(sims[i]) for i in picked], reverse=True)
            top1 = top_scores[0] if top_scores else 0.0
            top3mean = float(np.mean(top_scores[:3])) if top_scores else 0.0

            if (top1 < args.threshold_top) or (top3mean < args.threshold_mean):
                console.print("[yellow]No sé[/yellow]")
                continue

            ctx_rows = [meta[i] for i in picked]
            ctx = format_context(ctx_rows)
            sys_prompt = make_system_prompt(args.locale)
            user_prompt = make_user_prompt(q, ctx)
            ans = ollama_chat(args.model, sys_prompt, user_prompt, base_url=args.ollama_url)

            # Mostrar respuesta + fuentes
            console.print(f"[green]Asistente[/green]: {ans}\n")
            console.print("[dim]Fuentes:[/dim]")
            for r in ctx_rows:
                t0 = seconds_to_hhmmss(r["start"]); t1 = seconds_to_hhmmss(r["end"])
                console.print(f" - {r['folder']}/{r['video_name']} {t0}-{t1} [{r.get('speaker') or 'DESCONOCIDO'}]")

        except KeyboardInterrupt:
            console.print("\n[red]Fin[/red]")
            break

def main():
    ap = argparse.ArgumentParser(description="RAG local sobre entrevistas: index y chat (cerrado al corpus).")
    sp = ap.add_subparsers(dest="cmd", required=True)

    a = sp.add_parser("index", help="Construye el índice FAISS desde _corpus_segments.jsonl")
    a.add_argument("--corpus", required=True, help="Ruta a _corpus_segments.jsonl")
    a.add_argument("--out", default="index", help="Carpeta de salida del índice")
    a.add_argument("--embed-model", default="intfloat/multilingual-e5-base", help="Modelo de embeddings (multilingüe)")
    a.add_argument("--show-stats", action="store_true")
    a.set_defaults(func=cmd_index)

    c = sp.add_parser("chat", help="Inicia el chat local con Ollama (cerrado al corpus)")
    c.add_argument("--model", default="llama3.1:8b", help="Modelo de Ollama")
    c.add_argument("--index", default="index", help="Carpeta donde está el índice")
    c.add_argument("--k", type=int, default=8, help="Nº de segmentos a usar como contexto")
    c.add_argument("--threshold-top", type=float, default=0.35, help="Mínimo coseno top-1")
    c.add_argument("--threshold-mean", type=float, default=0.33, help="Mínimo media top-3")
    c.add_argument("--locale", default="es-CO", help="Variante regional para el estilo de respuesta")
    c.add_argument("--ollama-url", default="http://localhost:11434", help="URL del servidor de Ollama")
    c.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
