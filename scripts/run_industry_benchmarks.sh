#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
OUT="${1:-$ROOT/benchmark_results}"
mkdir -p "$OUT"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"
ctest --test-dir "$BUILD" --output-on-failure
{
  echo "timestamp_utc,$(date -u +%FT%TZ)"
  echo "uname,$(uname -srmo)"
  echo "compiler,$(c++ --version | head -1)"
  echo "cpus,$(nproc)"
  echo "cpu_model,$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
  echo "memory,$(free -h | awk '/Mem:/ {print $2}')"
} > "$OUT/system.csv"

echo 'kernel,rows,cols,threads,mean_s,sd_s,gflop_s,weight_gib_s' > "$OUT/kernel.csv"
for spec in '2048 2048 1' '2048 2048 5' '4096 4096 1' '4096 4096 5' '11008 4096 1' '11008 4096 5'; do
  read -r rows cols threads <<< "$spec"
  "$BUILD/memvanta_kernel_bench" --rows "$rows" --cols "$cols" --threads "$threads" --reps 10 | tail -n +2 >> "$OUT/kernel.csv"
done

echo 'kernel,rows,cols,threads,mean_s,sd_s,gflop_s,weight_gib_s' > "$OUT/thread_scaling.csv"
for t in 1 2 3 4 5; do
  "$BUILD/memvanta_kernel_bench" --rows 4096 --cols 4096 --threads "$t" --reps 10 | awk 'NR>1 && ($1 ~ /^q8_0,/ || $1 ~ /^q4_0,/) {print}' >> "$OUT/thread_scaling.csv"
done

echo "Results: $OUT"
