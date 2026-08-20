#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?usage: run_v07_headline.sh MODEL.gguf [THREADS] [LLAMA_BIN_DIR] [CORPUS_TXT]}
THREADS=${2:-$(nproc)}
LLAMA_BIN=${3:-}
CORPUS=${4:-}
OUT=${OUT:-v07_headline}
REPS=${REPS:-5}
BATCH=${BATCH:-64}
KV=${KV:-f16}
mkdir -p "$OUT"
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'; uname -a; lscpu; free -h; sha256sum "$MODEL"; stat -c 'bytes=%s' "$MODEL";
} > "$OUT/environment.txt"
/usr/bin/time -v ./build-v07/memvanta_real_bench --model "$MODEL" --threads "$THREADS" --ctx 1024 --prompt 512 --gen 128 --batch "$BATCH" --kv "$KV" --warmup 1 --reps "$REPS" --csv "$OUT/memvanta.csv" > "$OUT/memvanta.txt" 2> "$OUT/memvanta.time.txt"
./build-v07/memvanta_real --model "$MODEL" --threads "$THREADS" --ctx 1024 --temperature 0 --n 32 --prompt 'Once upon a time' > "$OUT/memvanta.greedy.txt"
if [[ -n "$CORPUS" ]]; then ./build-v07/memvanta_eval --model "$MODEL" --threads "$THREADS" --ctx 1024 --max-tokens 512 --text "$CORPUS" > "$OUT/memvanta.eval.txt"; fi
if [[ -n "$LLAMA_BIN" && -x "$LLAMA_BIN/llama-bench" ]]; then
  /usr/bin/time -v "$LLAMA_BIN/llama-bench" -m "$MODEL" -p 512 -n 128 -t "$THREADS" -r "$REPS" -ngl 0 -o json > "$OUT/llama-bench.json" 2> "$OUT/llama-bench.time.txt"
fi
if [[ -n "$LLAMA_BIN" && -x "$LLAMA_BIN/llama-cli" ]]; then
  "$LLAMA_BIN/llama-cli" -m "$MODEL" -ngl 0 -t "$THREADS" -c 1024 -p 'Once upon a time' -n 32 --temp 0 --seed 42 > "$OUT/llama.greedy.txt" 2> "$OUT/llama.greedy.stderr.txt" || true
fi
if [[ -n "$CORPUS" && -n "$LLAMA_BIN" && -x "$LLAMA_BIN/llama-perplexity" ]]; then
  "$LLAMA_BIN/llama-perplexity" -m "$MODEL" -ngl 0 -t "$THREADS" -f "$CORPUS" > "$OUT/llama.ppl.txt" 2> "$OUT/llama.ppl.stderr.txt" || true
fi
echo "results: $OUT"
