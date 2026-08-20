#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
MODEL="${1:-/tmp/memvanta-bench-512m.bin}"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"
ctest --test-dir "$BUILD" --output-on-failure

if [[ ! -f "$MODEL" ]]; then
  dd if=/dev/zero of="$MODEL" bs=8M count=64 status=progress
  sync
fi

echo '# Kernel benchmark: llama-bench-style warmup + repeated mean/stddev'
"$BUILD/memvanta_kernel_bench" --rows 4096 --cols 4096 --threads "$(nproc)" --reps 7

echo '# Streaming benchmark'
for pf in 0 2 4; do
  "$BUILD/memvanta" run "$MODEL" --chunk 32M --zero-copy --prefetch "$pf"
done
"$BUILD/memvanta" run "$MODEL" --chunk 32M --cache 128M --prefetch 0
"$BUILD/memvanta" run "$MODEL" --chunk 32M --cache 128M --prefetch 2
