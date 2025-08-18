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
    """
    Embebe garantizando que todo sea str y normalizando embeddings.
    Evita TypeError: TextEncodeInput...
    """
    vecs: List[np.ndarray] = []
    # dimensión segura si la necesitamos (ST>=3 expone este método)
    dim = getattr(encoder, "get_sentence_embedding_dimension", lambda: 768)()

    n = len(texts)
    for i in tqdm(range(0, n, batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]

        # --- sanitizar entradas a str ---
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
        )
        part = np.asarray(part, dtype="float32")
        vecs.append(part)

    if not vecs:
        return np.zeros((0, int(dim)), dtype="float32")
    return np.vstack(vecs)


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
def build_index(corpus_path: Path, out_dir: Path, embed_model: str) -> None:
    """
    Construye un índice FAISS (coseno vía IP con embeddings normalizados) a partir de _corpus_segments.jsonl.

    Entradas:
      - corpus_path: ruta al JSONL global agregado por el pipeline de transcripción.
      - out_dir: carpeta donde se guardarán index.faiss, meta.jsonl y encoder.json
      - embed_model: nombre del modelo de sentence-transformers (p.ej. 'intfloat/multilingual-e5-base').

    Salidas (en out_dir):
      - index.faiss   : índice vectorial FAISS (IndexFlatIP).
      - meta.jsonl    : metadatos por segmento (sin embeddings).
      - encoder.json  : información del encoder (modelo, dimensión, cantidad, flags).
    """
    import time
    import numpy as np
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus no encontrado: {corpus_path}")

    data = read_jsonl(corpus_path)
    if not data:
        raise ValueError(f"Corpus vacío o ilegible: {corpus_path}")

    # Detectar si es un encoder tipo E5 para aplicar prefijos recomendados
    is_e5 = "e5" in (embed_model or "").lower()

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    # Construcción de textos y metadatos
    for i, r in enumerate(data):
        seg_txt = (r.get("text") or "").strip()
        if not seg_txt:
            continue  # omitir segmentos vacíos

        spk = r.get("speaker")
        base_text = f"[{spk}] {seg_txt}" if spk else seg_txt
        # Prefijo para E5 (mejora recall en recuperación densa)
        if is_e5:
            base_text = f"passage: {base_text}"

        texts.append(base_text)
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

    if not texts:
        raise ValueError("No se encontraron segmentos no vacíos para indexar.")

    # Cargar encoder y generar embeddings normalizados
    encoder = load_encoder(embed_model)
    vecs = embed_texts(encoder, texts)  # np.ndarray [N, d] float32 normalizado

    if vecs.ndim != 2 or vecs.shape[0] != len(metas):
        raise RuntimeError("Dimensiones de embeddings no coinciden con metadatos.")

    d = int(vecs.shape[1])

    # Índice FAISS: IP con embeddings normalizados => coseno
    index = faiss.IndexFlatIP(d)
    index.add(vecs)

    # Guardar índice y metadatos
    faiss.write_index(index, str(out_dir / "index.faiss"))
    write_jsonl(out_dir / "meta.jsonl", metas)

    encoder_info = {
        "model": embed_model,
        "dim": d,
        "count": int(index.ntotal),
        "normalized": True,
        "is_e5": bool(is_e5),
        "created_at": int(time.time())
    }
    (out_dir / "encoder.json").write_text(
        json.dumps(encoder_info, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


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

def expand_context(meta, seed_rows, window_s: float, max_total: int = 60):
    """
    Dado un conjunto de filas meta 'semilla' (los hits), agrega vecinos por (mismo video)
    cuya ventana de tiempo caiga dentro de [start-window_s, end+window_s].
    Limita el total de segmentos para no desbordar el prompt.
    """
    selected = {(r["video_relpath"], r["start"], r["end"]) for r in seed_rows}
    out = list(seed_rows)
    for r in seed_rows:
        vrel, a, b = r["video_relpath"], float(r["start"]), float(r["end"])
        lo, hi = a - window_s, b + window_s
        # barrido simple (corpus típico no tiene millones de segmentos)
        for m in meta:
            if m["video_relpath"] != vrel:
                continue
            s, e = float(m["start"]), float(m["end"])
            if (vrel, s, e) in selected:
                continue
            if (s >= lo and s <= hi) or (e >= lo and e <= hi):
                out.append(m)
                selected.add((vrel, s, e))
                if len(out) >= max_total:
                    return sorted(out, key=lambda x: (x["folder"], x["video_name"], x["start"]))
    return sorted(out, key=lambda x: (x["folder"], x["video_name"], x["start"]))

def filter_candidates_by_person(candidates_ids, meta, person: str):
    if not person:
        return candidates_ids
    return [i for i in candidates_ids if (meta[i].get("folder") or "").lower() == person.lower()]


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
        "Eres un asistente que responde ÚNICAMENTE con base en el CONTEXTO proporcionado. "
        "Tu tarea es integrar y sintetizar ideas de múltiples fragmentos para responder de forma clara. "
        "Si el contexto no contiene la respuesta, escribe exactamente: 'No sé'. "
        "No inventes datos. No listes fuentes a menos que te lo pidan explícitamente. "
        "Responde en español, registro neutro de Colombia."
    )

def make_user_prompt(question: str, context: str) -> str:
    return (
        f"Pregunta: {question}\n\n"
        f"CONTEXTO:\n{context}\n\n"
        "Instrucción: Responde con la mejor síntesis posible usando solo el CONTEXTO. "
        "Si el contexto no contiene la respuesta, escribe exactamente 'No sé'."
    )

# ---------- CLI ----------
def print_sources(rows, max_sources: int, show: bool):
    if not show:
        return
    from itertools import islice
    print("\nFuentes:")
    for r in islice(rows, max_sources):
        t0 = seconds_to_hhmmss(r["start"]); t1 = seconds_to_hhmmss(r["end"])
        who = r.get("speaker") or "DESCONOCIDO"
        print(f" - {r['folder']}/{r['video_name']} {t0}-{t1} [{who}]")


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
    Chat RAG local, cerrado al corpus:
    - Búsqueda en FAISS (coseno con embeddings normalizados)
    - Filtro opcional por persona (--person)
    - MMR para diversidad
    - Ventanas alrededor de hits (--window-s) para coherencia
    - Gating por umbrales
    - LLM local vía /api/generate (Ollama 0.11.x)
    """
    index, meta, enc_info = load_index(Path(args.index))
    encoder = load_encoder(enc_info["model"])

    console.print("[bold]Chat RAG local[/bold] — escribe tu pregunta (Ctrl+C para salir)")
    while True:
        try:
            q = console.input("[cyan]Tú[/cyan]: ").strip()
            if not q:
                continue

            # Prefijo E5 para consultas (mejora recall)
            q_input = q
            enc_name = (enc_info.get("model") or "").lower()
            if "e5" in enc_name and not q.lower().startswith("query:"):
                q_input = f"query: {q}"

            # 1) Embedding de la consulta
            q_vec = embed_texts(encoder, [q_input])  # (1,d)

            # 2) Candidatos (fetch_k > k)
            fetch_k = min(args.k * 6, 150)
            D, I = index.search(q_vec, fetch_k)
            candidates = [int(i) for i in I[0].tolist() if i != -1]

            # 2b) Filtro por persona (carpeta), si aplica
            candidates = filter_candidates_by_person(candidates, meta, args.person)
            if not candidates:
                console.print("[yellow]No sé[/yellow]")
                continue

            # 3) MMR con reconstrucción de candidatos
            picked, sims_map = mmr_from_index(index, q_vec[0], candidates, topn=args.k, lambda_=0.5)

            # 4) Gating
            top_scores = sorted([float(sims_map[i]) for i in picked], reverse=True)
            top1 = top_scores[0] if top_scores else 0.0
            top3mean = float(np.mean(top_scores[:3])) if top_scores else 0.0
            if (top1 < args.threshold_top) or (top3mean < args.threshold_mean):
                console.print("[yellow]No sé[/yellow]")
                continue

            # 5) Expansión de contexto con ventanas por video
            seed_rows = [meta[i] for i in picked]
            ctx_rows = expand_context(meta, seed_rows, window_s=args.window_s, max_total=max(args.k*5, 60))

            # 6) Construcción de prompts y llamada a LLM
            ctx = format_context(ctx_rows)
            sys_prompt = make_system_prompt(args.locale)
            user_prompt = make_user_prompt(q, ctx)
            try:
                ans = ollama_chat(args.model, sys_prompt, user_prompt, base_url=args.ollama_url)
            except Exception as e:
                console.print(f"[red]Error Ollama:[/red] {e}")
                continue

            # 7) Salida
            console.print(f"[green]Asistente[/green]: {ans}")
            print_sources(ctx_rows, max_sources=args.max_sources, show=args.show_sources)
            print()

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
    c.add_argument("--locale", default="es-CO", help="Variante regional para el estilo de respuesta")
    c.add_argument("--ollama-url", default="http://localhost:11434", help="URL del servidor de Ollama")
    c.add_argument("--show-sources", action="store_true", help="Muestra/oculta la lista de fuentes en consola.")
    c.add_argument("--max-sources", type=int, default=4, help="Máximo de fuentes a mostrar si --show-sources.")
    c.add_argument("--window-s", type=float, default=8.0, help="Ventana (s) para expandir contexto alrededor de cada hit.")
    c.add_argument("--person", type=str, default=None, help="Filtra por carpeta/persona (ej. 'Alejo').")
    c.add_argument("--k", type=int, default=12, help="Segmentos base a usar (sube para más contexto agregado).")
    c.add_argument("--threshold-top", type=float, default=0.30, help="Mínimo coseno top-1 (bajar si dice 'No sé' muy seguido).")
    c.add_argument("--threshold-mean", type=float, default=0.28, help="Mínimo media top-3.")
    c.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
