#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$(nproc)"
ctest --test-dir "$ROOT/build" --output-on-failure
"$ROOT/build/memvanta_llama_bench" --threads "${THREADS:-$(nproc)}" --reps "${REPS:-5}" --warmup 1 --csv "${OUT:-$ROOT/v04_industry_results.csv}"
