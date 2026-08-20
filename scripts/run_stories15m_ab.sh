#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-stories15M-q4_0.gguf}
THREADS=${THREADS:-$(nproc)}
LLAMA_BIN=${LLAMA_BIN:-./llama.cpp/build/bin}
OUT=${OUT:-v072_stories15m_ab}
REPS=${REPS:-5}
WARMUP=${WARMUP:-1}
PROMPT=${PROMPT:-64}
GEN=${GEN:-64}
CTX=${CTX:-128}
BATCH=${BATCH:-32}
KV=${KV:-f16}
MEMVANTA_TIMEOUT=${MEMVANTA_TIMEOUT:-1800}
LLAMA_TIMEOUT=${LLAMA_TIMEOUT:-900}

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
  echo "threads=$THREADS prompt=$PROMPT gen=$GEN ctx=$CTX batch=$BATCH kv=$KV reps=$REPS warmup=$WARMUP"
  echo "memvanta_timeout_s=$MEMVANTA_TIMEOUT llama_timeout_s=$LLAMA_TIMEOUT"
  echo "client_workload=disabled"
  uname -a
  lscpu
  free -h
} > "$OUT/environment.txt"

status=0

echo "[benchmark] starting MemVanta at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee "$OUT/progress.txt"
set +e
timeout --signal=TERM --kill-after=30s "$MEMVANTA_TIMEOUT" \
  /usr/bin/time -v ./build-v072/memvanta_real_bench \
    --model "$MODEL" --threads "$THREADS" --ctx "$CTX" \
    --prompt "$PROMPT" --gen "$GEN" --batch "$BATCH" --kv "$KV" \
    --warmup "$WARMUP" --reps "$REPS" --no-client --csv "$OUT/memvanta.csv" \
    > "$OUT/memvanta.stdout.txt" 2> "$OUT/memvanta.time.txt"
mem_rc=$?
set -e
echo "[benchmark] MemVanta rc=$mem_rc finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/progress.txt"
echo "$mem_rc" > "$OUT/memvanta.exitcode.txt"
if [[ $mem_rc -ne 0 ]]; then status=1; fi

set +e
timeout --signal=TERM --kill-after=15s 180 \
  ./build-v072/memvanta_real --model "$MODEL" --threads "$THREADS" --ctx "$CTX" \
    --temperature 0 --n 8 --prompt 'Once upon a time' \
    > "$OUT/memvanta.greedy.txt" 2> "$OUT/memvanta.greedy.stderr.txt"
greedy_rc=$?
set -e
echo "$greedy_rc" > "$OUT/memvanta.greedy.exitcode.txt"
echo "[benchmark] MemVanta greedy rc=$greedy_rc" | tee -a "$OUT/progress.txt"
if [[ $greedy_rc -ne 0 ]]; then status=1; fi

if [[ -x "$LLAMA_BIN/llama-bench" ]]; then
  echo "[benchmark] starting llama.cpp at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/progress.txt"
  set +e
  timeout --signal=TERM --kill-after=15s "$LLAMA_TIMEOUT" \
    /usr/bin/time -v "$LLAMA_BIN/llama-bench" -m "$MODEL" -p "$PROMPT" -n "$GEN" \
      -t "$THREADS" -r "$REPS" -ngl 0 -o json \
      > "$OUT/llama-bench.json" 2> "$OUT/llama-bench.time.txt"
  llama_rc=$?
  set -e
  echo "$llama_rc" > "$OUT/llama-bench.exitcode.txt"
  echo "[benchmark] llama.cpp rc=$llama_rc finished at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "$OUT/progress.txt"
  if [[ $llama_rc -ne 0 ]]; then status=1; fi
else
  echo "llama-bench not found at $LLAMA_BIN/llama-bench" | tee -a "$OUT/progress.txt" >&2
  echo "127" > "$OUT/llama-bench.exitcode.txt"
  status=1
fi

if [[ -x "$LLAMA_BIN/llama-cli" ]]; then
  set +e
  timeout --signal=TERM --kill-after=15s 120 \
    "$LLAMA_BIN/llama-cli" -m "$MODEL" -ngl 0 -t "$THREADS" -c "$CTX" \
      -p 'Once upon a time' -n 8 --temp 0 --seed 42 \
      > "$OUT/llama.greedy.txt" 2> "$OUT/llama.greedy.stderr.txt"
  llama_greedy_rc=$?
  set -e
  echo "$llama_greedy_rc" > "$OUT/llama.greedy.exitcode.txt"
  if [[ $llama_greedy_rc -ne 0 ]]; then status=1; fi
else
  echo "127" > "$OUT/llama.greedy.exitcode.txt"
  status=1
fi

python3 scripts/summarize_ab.py "$OUT" || status=1
echo "results: $OUT"
exit "$status"
