#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?usage: run_real_compare.sh model.gguf [threads] [llama-bench]}
THREADS=${2:-$(nproc)}
LLAMA_BENCH=${3:-llama-bench}
OUT=${OUT:-real_compare}
REPS=${REPS:-5}
mkdir -p "$OUT"

# Recommended public reference checkpoint:
# QuantFactory/SmolLM2-135M-Instruct-GGUF / SmolLM2-135M-Instruct.Q4_0.gguf
# SHA256 f68203bfb98b1b1e8c64fac75fab10c4e36acac081609573b4df0fcc19c90dd9
sha256sum "$MODEL" | tee "$OUT/model.sha256"

./build-real/memvanta_real_bench \
  --model "$MODEL" --threads "$THREADS" --ctx 1024 \
  --prompt 512 --gen 128 --warmup 1 --reps "$REPS" --no-client \
  --csv "$OUT/memvanta.csv" | tee "$OUT/memvanta.md"

if command -v "$LLAMA_BENCH" >/dev/null 2>&1 || [[ -x "$LLAMA_BENCH" ]]; then
  "$LLAMA_BENCH" -m "$MODEL" -p 512 -n 128 -t "$THREADS" -r "$REPS" -dev none -ngl 0 -o json \
    | tee "$OUT/llama-bench.json"
else
  echo "llama-bench not found; pass its path as argument 3" >&2
fi
