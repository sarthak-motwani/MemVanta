# MemVanta CPU v0.7.2

[![Build and Test](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![Real Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml)
[![Medium Model Validation](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-validation.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-validation.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**MemVanta — Memory-Adaptive LLM Inference Beyond RAM and VRAM.**

A CPU-first C++20 inference runtime for GGUF models, quantized execution, hierarchical memory, batched prefill, compact KV caches, tokenizer parity testing, and reproducible end-to-end benchmarking.

## Project status

MemVanta is an engineering prototype under active development. The native inference stack, quantized kernels, tokenizer tests, benchmark harnesses, and CI are implemented.

The trained-model validation gates are green on `main`: the repeated TinyStories 15M Q4_0 same-model A/B workflow completes under the declared pp64/tg64 protocol, and the SmolLM2-360M Q4_0 validation gate completes tokenizer, greedy-generation, and real-model benchmark execution. This establishes successful native trained-model execution across both checkpoints.

The repeated TinyStories A/B now provides a quantitative trained-model result: MemVanta used **30,676 KiB peak RSS versus 43,204 KiB for pinned llama.cpp, a 29.0% reduction**, while llama.cpp remained substantially faster on this small-model CPU workload. This is a memory-efficiency result, not a throughput win.

## Verified trained-model results

### TinyStories 15M Q4_0 — repeated same-model A/B

CPU-only, 4 threads, context 128, pp64/tg64, batch 32, F16 KV, one warm-up plus five measured repetitions. MemVanta commit `d068fee3a7b3cac6cf079a01a556d3f3838c9925`; pinned llama.cpp commit `6503355df0eb4f65875012523263c302fe0088c1`.

| Metric | MemVanta | llama.cpp | Result |
|---|---:|---:|---:|
| Prompt processing | 2107.44 ± 43.81 tok/s | 8211.27 ± 428.90 tok/s | llama.cpp 3.90× faster |
| Token generation | 730.65 ± 76.04 tok/s | 2007.99 ± 115.31 tok/s | llama.cpp 2.75× faster |
| Peak RSS | 30,676 KiB | 43,204 KiB | **MemVanta 29.0% lower** |

MemVanta prompt-processing CV was 2.08%; token-generation CV was 10.41%, so the generation result carries a high-variance warning. Deterministic greedy validation passed and both benchmark processes exited successfully. Raw evidence is committed under `results/stories15m-repeated-ab/`.

### SmolLM2-360M Q4_0 — native CPU validation

On the 361.82M-parameter Q4_0 checkpoint with 4 CPU threads, MemVanta completed tokenizer, deterministic generation, and benchmark validation. The short benchmark reported **59.04 pp tok/s**, **34.42 tg tok/s**, **244.32 MiB peak RSS**, **22.70 ms load time**, and **5.00 MiB KV allocation**. This gate validates larger trained-model execution; it is not a llama.cpp comparison. Raw evidence is committed under `results/smollm2-360m/`.

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
7. uploads raw benchmark evidence that is automatically published under `results/` after a successful run.

The `Medium Model Validation` workflow separately validates SmolLM2-360M Q4_0 with tokenizer, deterministic greedy generation, and a short real-model benchmark. Successful artifacts are also automatically published under `results/`.

See `results/stories15m-repeated-ab/`, `results/smollm2-360m/`, `results/VALIDATION_GATES_2026-08-21.md`, and `V072_REAL_MODEL_AB.md` for the preserved evidence and protocol.

## Tokenizer hardening

v0.7.1 added hardened GGUF tokenization for GPT-2 byte-BPE and Llama/SentencePiece-style unigram tokenizers, including arbitrary-byte fallback, malformed UTF-8 handling, special-token recognition, tokenizer CLI tooling, deterministic fixtures, fuzz tests, and SentencePiece reference-parity tests. See `V071_TOKENIZER_HARDENING.md` for the executed test matrix and remaining parity gates.

## Contributing and security

Contributions are welcome. See `CONTRIBUTING.md` for build, test, and benchmark-evidence expectations. Please follow `SECURITY.md` for vulnerability reports.

## License

Licensed under the Apache License 2.0. See `LICENSE`.
