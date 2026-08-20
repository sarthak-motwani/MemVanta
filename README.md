# MemVanta CPU v0.7.2

**MemVanta — Memory-Adaptive LLM Inference Beyond RAM and VRAM.**

A CPU-first C++ inference runtime for GGUF models, quantized execution, hierarchical memory, batched prefill, compact KV caches, tokenizer parity testing, and reproducible end-to-end benchmarking.

v0.6 adds a register-blocked Q4/Q8 GEMM prefill path, optional once-per-batch Q8 activation quantization, SIMD F16/Q8 KV attention, precomputed allocation-free RoPE, persistent prefill workers, and an automatic batch/thread tuner.

## Project rename

The runtime is now named **MemVanta**. All project-owned namespaces, headers, CMake targets, executables, environment variables, benchmark scripts, workflow labels, and documentation use `memvanta` / `MEMVANTA`. Model-family names such as SmolLM2 and TinyLlama are unchanged because they refer to external models.

## Headline benchmark

On the same 5-vCPU AMD EPYC environment and the exact v0.5 primary settings (batch 64, FP32 KV, 5 threads, pp512/tg128, five repetitions + warm-up):

- v0.5 pp512: 133.60 ± 18.85 tok/s
- **v0.6 pp512: 322.42 ± 32.64 tok/s — 2.41× faster**
- v0.5 tg128: 67.57 ± 4.66 tok/s
- **v0.6 tg128: 67.79 ± 5.94 tok/s — decode preserved**

These numbers are from the same full-transformer SmolLM2-shaped synthetic GGUF systems workload; they are not a trained-model comparison. See `V06_GEMM_BENCHMARK.md` for methodology and limitations.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Benchmark

```bash
./build/memvanta_real_bench \
  --model model.gguf \
  --threads 5 --batch 64 --kv f32 \
  --prompt 512 --gen 128 --reps 5 --warmup 1
```

## Auto-tune

```bash
./build/memvanta_auto_tune --model model.gguf --max-threads 5 --prompt 128
```

## Optional Q8 activation path

```bash
MEMVANTA_V06_Q8_ACT=1 ./build/memvanta_real_bench ...
```

The default register-blocked activation path is faster on the benchmark host; Q8 activation quantization is retained for CPUs where it wins.

## Real-model A/B gate

The included GitHub Actions workflow downloads the trained TinyStories 15M Q4_0 GGUF, verifies its exact SHA-256, builds current llama.cpp CPU tools, and runs the same model through MemVanta and llama.cpp under the same CPU settings. Until that workflow completes successfully, this repository does **not** claim that MemVanta beats llama.cpp or that the trained-model parity gate has passed.

## v0.7.1 tokenizer hardening

v0.7.1 adds hardened GGUF tokenization for GPT-2 byte-BPE and Llama/SentencePiece-style unigram tokenizers, including arbitrary-byte fallback, malformed UTF-8 handling, special-token recognition, tokenizer CLI tooling, deterministic fixtures, fuzz tests, and SentencePiece reference-parity tests. See `V071_TOKENIZER_HARDENING.md` for the executed test matrix and remaining real-model parity gate.
