
# Chat RAG local — Entrevistas (100% offline en tiempo de ejecución)

Crea un chat que **solo** responde usando el contenido de las entrevistas.
Si la pregunta **no** está cubierta por el corpus, responde exactamente: **“No sé”**.

* Indexa `_corpus_segments.jsonl` (generado por tu repo de transcripción).
* Recupera contexto con **FAISS** y **embeddings** locales.
* Llama a un **LLM en Ollama** usando solo `POST /api/generate`.
* Aplica **gating** por similitud para mantener el chat **cerrado** al corpus.

---

## 🧱 Requisitos

* Python 3.10–3.12
* Ollama instalado y ejecutándose localmente (ej. `ollama serve`)

  * Modelo sugerido: `llama3.1:8b-instruct` (o el que prefieras ya descargado)
* El archivo **`_corpus_segments.jsonl`** generado por el repositorio de transcripción

**requirements.txt**

```txt
faiss-cpu>=1.8.0,<2.0
sentence-transformers>=3.0,<3.2
numpy>=1.26,<3.0
tqdm>=4.66,<5.0
rich>=13.7,<14.0
requests>=2.31,<3.0
```

> Tiempo de ejecución **100% local** (no se hacen llamadas externas). La **primera** vez que uses el modelo de embeddings, se descarga a la caché local.

---

## 📦 Instalación

```bash
# dentro del repo RAG
python3 -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

---

## 🗂️ Estructura esperada

```
/RAG/
  ├─ rag_chat.py
  ├─ requirements.txt
  └─ index/                 # (se crea al indexar)

# De tu repo de transcripción (usarás esta ruta):
<root_entrevistas>/_corpus_segments.jsonl
```

Cada línea del `_corpus_segments.jsonl` debe tener, entre otros, campos como:

```json
{
  "video_relpath": "Persona_A/MVI_0001.MOV",
  "video_name": "MVI_0001.MOV",
  "folder": "Persona_A",
  "start": 44.12,
  "end": 49.87,
  "speaker": "ENTREVISTADO",
  "text": "Contenido del segmento…"
}
```

---

## ⚙️ Flujo

### 1) Indexar el corpus

```bash
python rag_chat.py index \
  --corpus /ruta/a/_corpus_segments.jsonl \
  --embed-model intfloat/multilingual-e5-base \
  --out index \
  --show-stats
```

Esto crea:

* `index/index.faiss` — índice vectorial (coseno vía IP con embeddings normalizados)
* `index/meta.jsonl` — metadatos de segmentos
* `index/encoder.json` — info del encoder (modelo, dimensión, cantidad)

### 2) Verificar modelo en Ollama

```bash
ollama pull llama3.1:8b-instruct   # si aún no lo tienes
```

### 3) Chatear (cerrado al corpus)

```bash
python rag_chat.py chat \
  --model llama3.1:8b-instruct \
  --index index \
  --k 8 \
  --threshold-top 0.35 \
  --threshold-mean 0.33 \
  --locale es-CO
```

* Si la similitud del top-1 o la media de top-3 es **baja**, responde **“No sé”**.
* Si hay contexto suficiente, responde y muestra **fuentes** (carpeta/video + rango de tiempo + hablante).

---

## 🧠 Cómo funciona el “cierre” (gating)

1. **Búsqueda** de candidatos con FAISS (coseno).
2. **MMR** sobre candidatos para diversidad (reconstruye solo los vectores necesarios del índice).
3. **Umbrales**:

   * `--threshold-top`: similitud mínima del **top-1** (ej. 0.35).
   * `--threshold-mean`: media mínima del **top-3** (ej. 0.33).
4. Si **no** se cumplen, respuesta fija: **“No sé”**.

> Ajusta los umbrales a tu corpus: si el chat dice “No sé” demasiado, bájalos; si “se cuela” respuesta no sustentada, súbelos.

---

## 🧩 CLI — Comandos y opciones

### `index`

```bash
python rag_chat.py index \
  --corpus /ruta/a/_corpus_segments.jsonl \
  --out index \
  --embed-model intfloat/multilingual-e5-base \
  --show-stats
```

* `--corpus` (obligatorio): ruta al JSONL global.
* `--out`: carpeta de índice (por defecto `index`).
* `--embed-model`: encoder multilingüe (por defecto `intfloat/multilingual-e5-base`).
* `--show-stats`: imprime tabla resumen.

### `chat`

```bash
python rag_chat.py chat \
  --model llama3.1:8b-instruct \
  --index index \
  --k 8 \
  --threshold-top 0.35 \
  --threshold-mean 0.33 \
  --locale es-CO \
  --ollama-url http://localhost:11434
```

* `--model`: nombre del modelo en Ollama.
* `--index`: carpeta con el índice FAISS y metadatos.
* `--k`: cantidad de segmentos en contexto.
* `--threshold-top`, `--threshold-mean`: umbrales de similitud (coseno).
* `--locale`: estilo de respuesta (solo texto; el cierre lo impone el gating).
* `--ollama-url`: URL del servidor de Ollama.

> Llamadas a Ollama **solo** por `POST /api/generate` (compatibles con Ollama 0.11.x).

---

## 🛠️ Solución de problemas

* **“No sé” en todo**: baja umbrales (`--threshold-top 0.30 --threshold-mean 0.28`) o aumenta `--k`.
* **Ollama 404 / chat**: este proyecto usa **solo** `/api/generate`. Asegúrate de que `ollama serve` esté activo y el modelo disponible.
* **Embeddings lentos la primera vez**: es normal; el modelo se cachea localmente.
* **Corpus vacío o ilegible**: confirma que `_corpus_segments.jsonl` existe y tiene líneas JSON válidas.

---

## 📐 Decisiones de diseño

* **FAISS IndexFlatIP** + **embeddings normalizados** ⇒ similitud coseno.
* **MMR** “ligero” (solo candidatos) para evitar cargar todos los vectores a RAM.
* **Prompt del LLM** exige: responde **solo** con el contexto; si no hay evidencia, **“No sé”**.
* **Runtime offline**: sin llamadas externas en el chat; Ollama y embeddings ya en disco.

---

## 🧪 Ejemplo mínimo

```bash
# 1) Crear venv e instalar deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Indexar
python rag_chat.py index \
  --corpus /ruta/entrevistas/_corpus_segments.jsonl \
  --out index \
  --show-stats

# 3) Asegurar modelo en Ollama
ollama pull llama3.1:8b-instruct

# 4) Chat
python rag_chat.py chat \
  --model llama3.1:8b-instruct \
  --index index \
  --k 8 \
  --threshold-top 0.35 \
  --threshold-mean 0.33 \
  --locale es-CO
```

---

## 📄 Licencia

MIT (o la que prefieras).

