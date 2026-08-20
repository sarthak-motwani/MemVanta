#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:-stories260K.bin}
TOK=${2:-tok512.bin}
THREADS=${THREADS:-1}
REPS=${REPS:-5}
BUILD=${BUILD:-build_v04_real}
LLAMA_CPP=${LLAMA_CPP:-../llama.cpp}

expected_model=b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696
expected_tok=e6e45b754b603ab1fb1a31e59c1ebbee92a789504c3ddf6debb6bd3c106222d6
printf '%s  %s\n' "$expected_model" "$MODEL" | sha256sum -c -
printf '%s  %s\n' "$expected_tok" "$TOK" | sha256sum -c -

cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j "$(nproc)"
ctest --test-dir "$BUILD" --output-on-failure

"$BUILD/memvanta_llama2c_bench" --model "$MODEL" --threads "$THREADS" --pp 512 --tg 128 --reps "$REPS" --warmup 1 | tee memvanta_stories260k.txt

CONVERTER="$LLAMA_CPP/build/bin/llama-convert-llama2c-to-ggml"
BENCH="$LLAMA_CPP/build/bin/llama-bench"
if [[ ! -x "$CONVERTER" || ! -x "$BENCH" ]]; then
  echo "llama.cpp converter/llama-bench not found under $LLAMA_CPP/build/bin" >&2
  exit 3
fi
"$CONVERTER" --copy-vocab-from-model "$TOK" --llama2c-model "$MODEL" --llama2c-output-model stories260K.gguf
"$BENCH" -m stories260K.gguf -t "$THREADS" -p 512 -n 128 -r "$REPS" -o csv > llama_cpp_stories260k.csv

echo "Wrote memvanta_stories260k.txt and llama_cpp_stories260k.csv"
