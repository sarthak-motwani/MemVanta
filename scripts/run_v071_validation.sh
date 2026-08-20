#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build-v071-validation}"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"${JOBS:-2}"
ctest --test-dir "$BUILD" --output-on-failure

echo "Tokenizer hardening validation passed."
if [[ $# -ge 1 ]]; then
  MODEL="$1"
  THREADS="${2:-4}"
  echo "Running optional real-GGUF smoke benchmark: $MODEL"
  "$BUILD/memvanta_real_bench" --model "$MODEL" --threads "$THREADS" --pp 32 --tg 8 --reps 1 --warmup 1
fi
