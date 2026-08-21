# MemVanta CPU v0.7.2

[![Build and Test](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![Real Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**MemVanta — Memory-Adaptive LLM Inference Beyond RAM and VRAM.**

A CPU-first C++20 inference runtime for GGUF models, quantized execution, hierarchical memory, batched prefill, compact KV caches, tokenizer parity testing, and reproducible end-to-end benchmarking.

## Project status

MemVanta is an engineering prototype under active development. The native inference stack, quantized kernels, tokenizer tests, benchmark harnesses, and CI are implemented.

The trained-model validation gates are now green on `main`: the repeated TinyStories 15M Q4_0 same-model A/B workflow completes under the declared pp64/tg64 protocol, and the SmolLM2-360M Q4_0 validation gate completes tokenizer, greedy-generation, and real-model benchmark execution. This establishes successful native trained-model execution across both checkpoints.

MemVanta does **not** claim to be faster than llama.cpp based on green status alone. Comparative throughput and memory claims must come from preserved repeated A/B numerical artifacts. See `results/VALIDATION_GATES_2026-08-21.md` and `V072_REAL_MODEL_AB.md`.

## Headline systems benchmark

On the same 5-vCPU AMD EPYC environment and the exact v0.5 primary settings (batch 64, FP32 KV, 5 threads, pp512/tg128, five repetitions plus warm-up):

- v0.5 pp512: 133.60 ± 18.85 tok/s
- **v0.6 pp512: 322.42 ± 32.64 tok/s — 2.41× faster**
- v0.5 tg128: 67.57 ± 4.66 tok/s
- **v0.6 tg128: 67.79 ± 5.94 tok/s — decode preserved**

These results are from the same full-transformer SmolLM2-shaped **synthetic GGUF systems workload**. They are not trained-model benchmark results. See `V06_GEMM_BENCHMARK.md` for methodology and limitations.

## Core capabilities

- Native C++20 runtime with direct GGUF parsing
- F32, F16, GGUF Q4_0, and Q8_0 execution paths
- AVX2/FMA quantized kernels and register-blocked prefill GEMM
- RMSNorm, RoPE, GQA attention, causal attention, and SwiGLU
- Paged KV cache with F32/F16/Q8 storage options
- GPT-2 byte-BPE and Llama/SentencePiece-style tokenizer support
- mmap-based tensor access, bounded caching, eviction, and prefetch support
- Persistent CPU worker pool, batched prefill, decode fast path, and auto-tuning
- Reproducible benchmark tooling and real-model A/B workflow

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

The default register-blocked activation path was faster on the original benchmark host; Q8 activation quantization is retained for CPUs where it wins.

## Real-model validation

The `Real Model A/B` GitHub Actions workflow:

1. builds MemVanta and runs CTest,
2. downloads and verifies a trained TinyStories 15M Q4_0 GGUF,
3. builds a pinned llama.cpp CPU reference,
4. runs both runtimes on the same GGUF and CPU allocation,
5. executes the repeated pp64/tg64 benchmark gate with one warm-up and five measured repetitions,
6. performs deterministic greedy-generation and profiling checks, and
7. uploads raw benchmark evidence.

The `Medium Model Validation` workflow separately validates SmolLM2-360M Q4_0 with tokenizer, deterministic greedy generation, and a short real-model benchmark.

Both validation gates are green on `main` as of 2026-08-21. See `results/VALIDATION_GATES_2026-08-21.md` for the execution-status record and `V072_REAL_MODEL_AB.md` for the repeated A/B protocol.

## Tokenizer hardening

v0.7.1 added hardened GGUF tokenization for GPT-2 byte-BPE and Llama/SentencePiece-style unigram tokenizers, including arbitrary-byte fallback, malformed UTF-8 handling, special-token recognition, tokenizer CLI tooling, deterministic fixtures, fuzz tests, and SentencePiece reference-parity tests. See `V071_TOKENIZER_HARDENING.md` for the executed test matrix and remaining parity gates.

## Contributing and security

Contributions are welcome. See `CONTRIBUTING.md` for build, test, and benchmark-evidence expectations. Please follow `SECURITY.md` for vulnerability reports.

## License

Licensed under the Apache License 2.0. See `LICENSE`.
