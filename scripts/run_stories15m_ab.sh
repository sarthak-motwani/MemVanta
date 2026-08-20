#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-stories15M-q4_0.gguf}
THREADS=${THREADS:-$(nproc)}
LLAMA_BIN=${LLAMA_BIN:-./llama.cpp/build/bin}
OUT=${OUT:-v072_stories15m_ab}
REPS=${REPS:-5}
PROMPT=${PROMPT:-64}
GEN=${GEN:-64}
CTX=${CTX:-128}
BATCH=${BATCH:-32}
KV=${KV:-f16}

mkdir -p "$OUT"
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 2
fi
if (( PROMPT + GEN > CTX )); then
  echo "PROMPT+GEN must be <= CTX for this native-context TinyStories checkpoint" >&2
  exit 2
fi

{
  echo "utc=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "model=$MODEL"
  sha256sum "$MODEL"
  stat -c 'model_bytes=%s' "$MODEL"
  echo "threads=$THREADS prompt=$PROMPT gen=$GEN ctx=$CTX batch=$BATCH kv=$KV reps=$REPS"
  uname -a
  lscpu
  free -h
} > "$OUT/environment.txt"

/usr/bin/time -v ./build-v072/memvanta_real_bench \
  --model "$MODEL" --threads "$THREADS" --ctx "$CTX" \
  --prompt "$PROMPT" --gen "$GEN" --batch "$BATCH" --kv "$KV" \
  --warmup 1 --reps "$REPS" --csv "$OUT/memvanta.csv" \
  > "$OUT/memvanta.stdout.txt" 2> "$OUT/memvanta.time.txt"

./build-v072/memvanta_real --model "$MODEL" --threads "$THREADS" --ctx "$CTX" \
  --temperature 0 --n 32 --prompt 'Once upon a time' \
  > "$OUT/memvanta.greedy.txt" 2> "$OUT/memvanta.greedy.stderr.txt" || true

if [[ -x "$LLAMA_BIN/llama-bench" ]]; then
  /usr/bin/time -v "$LLAMA_BIN/llama-bench" -m "$MODEL" -p "$PROMPT" -n "$GEN" \
    -t "$THREADS" -r "$REPS" -ngl 0 -o json \
    > "$OUT/llama-bench.json" 2> "$OUT/llama-bench.time.txt"
else
  echo "llama-bench not found at $LLAMA_BIN/llama-bench" >&2
fi

if [[ -x "$LLAMA_BIN/llama-cli" ]]; then
  "$LLAMA_BIN/llama-cli" -m "$MODEL" -ngl 0 -t "$THREADS" -c "$CTX" \
    -p 'Once upon a time' -n 32 --temp 0 --seed 42 \
    > "$OUT/llama.greedy.txt" 2> "$OUT/llama.greedy.stderr.txt" || true
fi

python3 scripts/summarize_ab.py "$OUT"
echo "results: $OUT"
