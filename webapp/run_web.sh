#!/usr/bin/env bash
set -euo pipefail

# --- ubicaciones ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # debe contener web_chat.py, rag_core.py, requirements.txt

# --- config por defecto (puedes exportarlas antes de ejecutar) ---
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${MODEL_NAME:=llama3.1:8b-instruct}"
: "${OLLAMA_URL:=http://localhost:11434}"

# INDEX_DIR por defecto: ../index (junto al repo, no dentro de webapp/)
DEFAULT_INDEX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/index"
: "${INDEX_DIR:=$DEFAULT_INDEX_DIR}"

echo "== Chat RAG Web: preparación del entorno =="
echo "Python: $(python3 --version || true)"

# --- venv ---
if [[ ! -d ".venv" ]]; then
  echo "Creando entorno virtual (.venv)..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# --- validar índice ---
REQ_FILES=( "index.faiss" "meta.jsonl" "encoder.json" )
MISSING=0
for f in "${REQ_FILES[@]}"; do
  if [[ ! -f "$INDEX_DIR/$f" ]]; then
    MISSING=1
  fi
done

if [[ "$MISSING" -eq 1 ]]; then
  echo "No se encontró un índice válido en: $INDEX_DIR"
  echo "Se esperaban: index.faiss, meta.jsonl, encoder.json"
  read -r -p "Ingresa la ruta a la carpeta del índice (o Enter para cancelar): " NEW_INDEX
  if [[ -z "${NEW_INDEX:-}" ]]; then
    echo "Operación cancelada."
    exit 1
  fi
  INDEX_DIR="$NEW_INDEX"
  MISSING=0
  for f in "${REQ_FILES[@]}"; do
    if [[ ! -f "$INDEX_DIR/$f" ]]; then
      MISSING=1
    fi
  done
  if [[ "$MISSING" -eq 1 ]]; then
    echo "La ruta proporcionada no contiene un índice válido. Abortando."
    exit 1
  fi
fi

# --- verificar Ollama ---
echo "Verificando Ollama en: $OLLAMA_URL ..."
if command -v curl >/dev/null 2>&1; then
  if ! curl -sS "${OLLAMA_URL}/api/tags" >/dev/null; then
    echo "No se pudo contactar a Ollama (${OLLAMA_URL})."
    echo "Asegúrate de tener 'ollama serve' en ejecución."
    exit 1
  fi
else
  echo "curl no está disponible; se omite verificación de Ollama."
fi

# --- exportar variables ---
export INDEX_DIR OLLAMA_URL MODEL_NAME

echo "== Listo. Lanzando servidor =="
echo "Host: $HOST  |  Port: $PORT"
echo "Modelo: $MODEL_NAME"
echo "Índice: $INDEX_DIR"
echo "URL: http://$HOST:$PORT"

exec uvicorn web_chat:app --host "$HOST" --port "$PORT"
