#!/usr/bin/env bash
set -euo pipefail

# start_and_query.sh
# Starts run_api.py and run_reader_api.py, waits until healthy, then sends a question
# Usage:
#  ./start_and_query.sh --retriever-args "--model_name MODEL --target_file data.txt --target_id_file data.id --target_embed_save_file embeddings" \
#                      --reader-args "--model_path CHECKPOINT --tokenizer_name TOKENIZER" \
#                      --question "What is ..."
# If --question is omitted, the script will prompt for one interactively.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RETR_SCRIPT="retriever/simcse/run_api.py"
READER_SCRIPT="generator/fid/run_reader_api.py"

RETR_ARGS=""
READER_ARGS=""
QUESTION=""
RETR_PORT=5000
READER_PORT=5001

print_usage(){
  sed -n '1,120p' "$0" | sed -n '1,40p'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --retriever-args)
      RETR_ARGS="$2"; shift 2;;
    --reader-args)
      READER_ARGS="$2"; shift 2;;
    --question)
      QUESTION="$2"; shift 2;;
    --retriever-port)
      RETR_PORT="$2"; shift 2;;
    --reader-port)
      READER_PORT="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

if [[ -z "$RETR_ARGS" ]]; then
  echo "Error: --retriever-args is required" >&2
  exit 1
fi
if [[ -z "$READER_ARGS" ]]; then
  echo "Error: --reader-args is required" >&2
  exit 1
fi

if [[ -z "$QUESTION" ]]; then
  read -rp $'Enter question: ' QUESTION
fi

mkdir -p "$ROOT_DIR/logs"

echo "Starting retriever: python $RETR_SCRIPT $RETR_ARGS --port $RETR_PORT"
conda run -n fftenv python "$RETR_SCRIPT" $RETR_ARGS --port $RETR_PORT > "$ROOT_DIR/logs/run_api.log" 2>&1 &
RETR_PID=$!

echo "Starting reader: python $READER_SCRIPT $READER_ARGS --port $READER_PORT"
conda run -n genenv python "$READER_SCRIPT" $READER_ARGS --port $READER_PORT > "$ROOT_DIR/logs/run_reader_api.log" 2>&1 &
READER_PID=$!

cleanup() {
  kill $RETR_PID 2>/dev/null || true
  kill $READER_PID 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for_health(){
  local url="$1"
  local timeout=${2:-30}
  local start=$(date +%s)
  while true; do
    if wget -qO- "$url" >/dev/null 2>&1; then
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout )); then
      echo "Timeout waiting for $url" >&2
      return 1
    fi
    sleep 1
  done
}

echo "Waiting for retriever health on http://localhost:$RETR_PORT/health"
wait_for_health "http://localhost:$RETR_PORT/health" 60
echo "Waiting for reader health on http://localhost:$READER_PORT/health"
wait_for_health "http://localhost:$READER_PORT/health" 60

echo "Sending question to retriever..."
PAYLOAD=$(conda run -n fftenv python -c "import json,sys; print(json.dumps({'question': sys.argv[1]}))" "$QUESTION")
echo "PAYLOAD: "
echo $PAYLOAD
RETR_JSON=$(wget -qO- --header="Content-Type: application/json" --post-data="$PAYLOAD" "http://localhost:$RETR_PORT/retrieve")

if [[ -z "$RETR_JSON" ]]; then
  echo "Retriever returned empty response" >&2
  exit 1
fi

echo "Retriever response:"
if command -v jq >/dev/null 2>&1; then
  echo "$RETR_JSON" | jq '.' || echo "$RETR_JSON"
elif python3 -c 'import sys,json; json.load(sys.stdin)' <<<"$RETR_JSON" >/dev/null 2>&1; then
  python3 -m json.tool <<<"$RETR_JSON" || echo "$RETR_JSON"
else
  echo "$RETR_JSON"
fi

echo "Forwarding to reader..."
READER_JSON=$(wget -qO- --header="Content-Type: application/json" --post-data="$RETR_JSON" "http://localhost:$READER_PORT/answer")

echo "Reader response:"
if command -v jq >/dev/null 2>&1; then
  echo "$READER_JSON" | jq '.' || echo "$READER_JSON"
elif python3 -c 'import sys,json; json.load(sys.stdin)' <<<"$READER_JSON" >/dev/null 2>&1; then
  python3 -m json.tool <<<"$READER_JSON" || echo "$READER_JSON"
else
  echo "$READER_JSON"
fi

echo "Done. Logs: $ROOT_DIR/logs/run_api.log and run_reader_api.log"
echo ""

CLEAN_CODE=$(jq -r 'def firstc: if type=="array" then .[0] else . end; if type=="array" then (.[0].clean_code | firstc) else (.clean_code | firstc) end // ""' <<<"$READER_JSON")
echo ""
echo "================================================================="
echo "Question: "
echo "$QUESTION"
echo ""
echo "Clean_code: "
echo "$CLEAN_CODE"
echo "================================================================="
echo ""


