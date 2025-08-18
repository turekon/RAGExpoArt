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

def mmr_from_index(index, query_vec: np.ndarray, candidate_ids: List[int], topn: int, lambda_: float = 0.5):
    """
    MMR usando únicamente los vectores de los candidatos (reconstruidos desde el índice).
    Requiere que los embeddings almacenados estén normalizados (lo están si usaste SentenceTransformers con normalize_embeddings=True).
    """
    # reconstruye los vectores de los candidatos
    cand_vecs = []
    for cid in candidate_ids:
        v = np.asarray(index.reconstruct(int(cid)), dtype="float32")
        cand_vecs.append(v)
    cand_vecs = np.vstack(cand_vecs) if cand_vecs else np.zeros((0, query_vec.shape[0]), dtype="float32")

    # similitud con la query (IP == coseno si todo normalizado)
    sims = (cand_vecs @ query_vec.reshape(-1, 1)).ravel()
    order = sims.argsort()[::-1].tolist()  # candidatos ordenados por similitud desc

    selected = []
    while order and len(selected) < topn:
        if not selected:
            selected.append(order.pop(0))
            continue
        best_j = None
        best_score = -1e9
        for j in order:
            sim_q = sims[j]
            # penalización por similitud con lo ya seleccionado
            sim_sel = max(float(cand_vecs[j] @ cand_vecs[s]) for s in selected)
            score = lambda_ * sim_q - (1.0 - lambda_) * sim_sel
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        order.remove(best_j)

    # devolver ids de índice global correspondientes y un mapa de similitudes
    picked_ids = [int(candidate_ids[j]) for j in selected]
    sim_map = {int(candidate_ids[j]): float(sims[j]) for j in range(len(candidate_ids))}
    return picked_ids, sim_map

def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta = read_jsonl(index_dir / "meta.jsonl")
    enc_info = json.loads((index_dir / "encoder.json").read_text(encoding="utf-8"))
    return index, meta, enc_info

# ---------- LLM (Ollama) ----------
def ollama_chat(model: str, system_prompt: str, user_prompt: str, base_url: str = "http://localhost:11434"):
    """
    Implementación fija para Ollama 0.11.x usando únicamente /api/generate.
    Combina system + user en un único prompt.
    """
    import requests

    url_gen = f"{base_url}/api/generate"
    prompt = (
        f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{user_prompt}"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 512
        },
        "stream": False
    }
    r = requests.post(url_gen, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

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
    """
    Inicia el chat RAG local:
    - Búsqueda en FAISS (IndexFlatIP con embeddings normalizados)
    - MMR para diversidad con reconstrucción de candidatos
    - Gating por umbrales (top-1 y media top-3)
    - Llamada a Ollama vía /api/generate (0.11.x)
    """
    index, meta, enc_info = load_index(Path(args.index))
    encoder = load_encoder(enc_info["model"])

    console.print("[bold]Chat RAG local[/bold] — escribe tu pregunta (Ctrl+C para salir)")
    while True:
        try:
            q = console.input("[cyan]Tú[/cyan]: ").strip()
            if not q:
                continue

            # 1) Embeddings de la consulta
            q_vec = embed_texts(encoder, [q])  # (1, d)

            # 2) Búsqueda inicial para candidatos
            fetch_k = min(args.k * 6, 100)
            D, I = index.search(q_vec, fetch_k)
            candidates = [int(i) for i in I[0].tolist() if i != -1]
            if not candidates:
                console.print("[yellow]No sé[/yellow]")
                continue

            # 3) MMR con reconstrucción de vectores de los candidatos
            picked, sims_map = mmr_from_index(index, q_vec[0], candidates, topn=args.k, lambda_=0.5)

            # 4) Gating por similitud (coseno)
            top_scores = sorted([float(sims_map[i]) for i in picked], reverse=True)
            top1 = top_scores[0] if top_scores else 0.0
            top3mean = float(np.mean(top_scores[:3])) if top_scores else 0.0

            if (top1 < args.threshold_top) or (top3mean < args.threshold_mean):
                console.print("[yellow]No sé[/yellow]")
                continue

            # 5) Construcción de contexto y prompt
            ctx_rows = [meta[i] for i in picked]
            ctx = format_context(ctx_rows)
            sys_prompt = make_system_prompt(args.locale)
            user_prompt = make_user_prompt(q, ctx)

            # 6) LLM local (Ollama /api/generate)
            try:
                ans = ollama_chat(args.model, sys_prompt, user_prompt, base_url=args.ollama_url)
            except Exception as e:
                console.print(f"[red]Error al llamar a Ollama:[/red] {e}")
                continue

            # 7) Respuesta y fuentes
            console.print(f"[green]Asistente[/green]: {ans}\n")
            console.print("[dim]Fuentes:[/dim]")
            for pid, r in zip(picked, ctx_rows):
                t0 = seconds_to_hhmmss(r["start"]); t1 = seconds_to_hhmmss(r["end"])
                who = r.get("speaker") or "DESCONOCIDO"
                console.print(f" - {r['folder']}/{r['video_name']} {t0}-{t1} [{who}]")

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
