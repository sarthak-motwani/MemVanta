# MemVanta CPU v0.7.2

[![Build and Test](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![Real Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml)
[![Medium Model Validation](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-validation.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-validation.yml)
[![Medium Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **Memory-adaptive LLM inference beyond conventional RAM and VRAM constraints.**

MemVanta is a CPU-first C++20 inference runtime for GGUF language models. It combines quantized execution, hierarchical memory management, compact KV caches, batched prefill, tokenizer support, and reproducible benchmark tooling in a systems-oriented runtime designed to explore how far useful LLM inference can be pushed under tight memory budgets.

The project is deliberately benchmark-driven: synthetic kernel results, trained-model validation, and direct llama.cpp comparisons are reported separately so performance claims remain auditable.

## Why MemVanta

Most inference engines optimize primarily for throughput on hardware where the model comfortably fits in available memory. MemVanta explores a different systems question:

**How efficiently can an LLM runtime execute when memory, not only compute, is the limiting resource?**

The current implementation focuses on:

- reducing resident memory pressure,
- executing quantized GGUF tensors directly,
- compact and pageable KV storage,
- mmap-based model access and bounded caching,
- CPU-efficient prefill and decode paths,
- deterministic trained-model validation, and
- reproducible same-model comparisons against established runtimes.

MemVanta is an engineering prototype under active development, not a claim of universal superiority over llama.cpp or other mature inference engines.

## Verified results

### TinyStories 15M Q4_0 — repeated MemVanta vs llama.cpp A/B

This is the current fully repeated same-model comparison.

**Protocol:** same trained GGUF, CPU-only, 4 threads, context 128, pp64/tg64, batch 32, F16 KV, one warm-up and five measured repetitions. The llama.cpp reference is pinned to commit `6503355df0eb4f65875012523263c302fe0088c1`.

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 2,107.44 ± 43.81 tok/s | 8,211.27 ± 428.90 tok/s | llama.cpp 3.90× faster |
| Token generation | 730.65 ± 76.04 tok/s | 2,007.99 ± 115.31 tok/s | llama.cpp 2.75× faster |
| Peak RSS | **30,676 KiB** | 43,204 KiB | **MemVanta 29.0% lower** |

The result is a **memory-efficiency win, not a throughput win**. MemVanta prompt-processing CV was 2.08%; generation CV was 10.41%, so the generation measurement carries a high-variance warning.

Raw evidence, per-run statistics, process exit codes, environment information, and profiling artifacts are committed under [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/).

### SmolLM2-360M Q4_0 — native trained-model validation

MemVanta also executes a trained **361.82M-parameter** SmolLM2 Q4_0 checkpoint natively on CPU.

| Metric | Result |
|---|---:|
| Prompt processing | **59.04 tok/s** |
| Token generation | **34.42 tok/s** |
| Peak RSS | **244.32 MiB** |
| Model load time | **22.70 ms** |
| KV allocation | **5.00 MiB** |

This workflow includes tokenizer validation, deterministic greedy generation, and real-model execution. The published result is a native-execution validation and should not be interpreted as a llama.cpp comparison.

Raw evidence is under [`results/smollm2-360m/`](results/smollm2-360m/).

### SmolLM2-360M same-model A/B

A dedicated `Medium Model A/B` workflow now extends the stricter comparison protocol to SmolLM2-360M using:

- the exact same Q4_0 GGUF and verified SHA-256,
- CPU-only execution,
- 4 threads,
- pp512 / tg128,
- context 768,
- batch 32,
- F16 KV,
- one warm-up plus five measured repetitions, and
- the same pinned llama.cpp reference.

Successful workflow evidence is published under `results/smollm2-360m-ab/`. Until a successful numerical artifact is published, no comparative 360M throughput or memory claim is made here.

## Performance evolution

MemVanta has improved substantially across releases while keeping synthetic and trained-model evidence clearly separated.

### v0.5 → v0.6 prefill improvement

On the same 5-vCPU AMD EPYC environment, same SmolLM2-shaped synthetic systems workload, batch 64, FP32 KV, five threads, pp512/tg128, and five measured repetitions:

| Version | pp512 | tg128 | Peak RSS |
|---|---:|---:|---:|
| v0.5 | 133.60 ± 18.85 tok/s | 67.57 ± 4.66 tok/s | ~124 MiB |
| v0.6 | **322.42 ± 32.64 tok/s** | **67.79 ± 5.94 tok/s** | 124.65 MiB |
| Change | **2.41× / +141.3%** | essentially unchanged | essentially unchanged |

These are **synthetic full-transformer systems results**, not trained-model benchmark results. See [`V06_GEMM_BENCHMARK.md`](V06_GEMM_BENCHMARK.md) for methodology and limitations.

## Core capabilities

- Native C++20 runtime with direct GGUF parsing
- F32, F16, GGUF Q4_0, and Q8_0 execution paths
- AVX2/FMA quantized kernels
- Register-blocked batched prefill GEMM
- RMSNorm, RoPE, GQA attention, causal attention, and SwiGLU
- Paged KV cache with F32, F16, and Q8 storage modes
- GPT-2 byte-BPE tokenizer support
- Llama/SentencePiece-style tokenizer support
- mmap-based tensor access
- Bounded caching, eviction, and prefetch support
- Persistent CPU worker pool
- Dedicated decode fast path
- Hardware-aware batch/thread auto-tuning
- Deterministic greedy-generation validation
- Reproducible benchmark and evidence-publication workflows

## Build

Requirements include a C++20 compiler and CMake.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run a real-model benchmark

```bash
./build/memvanta_real_bench \
  --model model.gguf \
  --threads 5 \
  --batch 64 \
  --kv f32 \
  --prompt 512 \
  --gen 128 \
  --reps 5 \
  --warmup 1
```

For benchmark claims, use repeated measurements rather than a single run and record the exact model SHA, CPU allocation, KV type, batch size, context, compiler, and MemVanta commit.

## Auto-tune

MemVanta can search thread and batch combinations instead of assuming that maximum parallelism is optimal on every CPU:

```bash
./build/memvanta_auto_tune \
  --model model.gguf \
  --max-threads 5 \
  --prompt 128
```

## Optional Q8 activation path

```bash
MEMVANTA_V06_Q8_ACT=1 ./build/memvanta_real_bench ...
```

The register-blocked FP32-activation path was faster on the original benchmark host, so it remains the default. The Q8 activation path is retained for CPUs and workloads where it benchmarks better.

## Benchmark methodology

MemVanta treats benchmark evidence as part of the implementation rather than as marketing output.

A valid same-model A/B comparison should keep constant:

1. the exact GGUF file and SHA-256,
2. quantization format,
3. CPU-only/GPU-offload policy,
4. logical CPU allocation and thread count,
5. context length,
6. prompt and generation workload,
7. KV format,
8. batch and micro-batch configuration, and
9. warm-up and repetition count.

Published workflows also preserve process RSS, environment details, exact runtime commits, raw outputs, and repeated throughput samples where available.

### Real Model A/B workflow

The `Real Model A/B` GitHub Actions workflow:

1. builds MemVanta and runs CTest,
2. downloads and SHA-verifies the trained model,
3. builds a pinned llama.cpp CPU reference,
4. runs both runtimes against the same GGUF,
5. records repeated prompt-processing and generation throughput,
6. captures peak RSS,
7. performs deterministic greedy-generation and profiling checks, and
8. publishes successful raw evidence under `results/`.

### Medium Model Validation

The `Medium Model Validation` workflow separately checks SmolLM2-360M tokenizer behavior, deterministic generation, native execution, timing, and peak RSS.

### Medium Model A/B

The `Medium Model A/B` workflow applies the repeated same-model comparison protocol to SmolLM2-360M at pp512/tg128. This is intended to test whether MemVanta's observed memory advantage persists as model size and workload increase.

## Evidence and reproducibility

Key evidence is kept in the repository rather than only in transient CI logs:

- [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/) — repeated TinyStories MemVanta/llama.cpp A/B
- [`results/smollm2-360m/`](results/smollm2-360m/) — SmolLM2-360M trained-model validation
- `results/smollm2-360m-ab/` — published 360M A/B evidence when available
- [`results/VALIDATION_GATES_2026-08-21.md`](results/VALIDATION_GATES_2026-08-21.md) — current validation boundary
- [`V072_REAL_MODEL_AB.md`](V072_REAL_MODEL_AB.md) — repeated A/B protocol
- [`V07_REAL_MODEL_HEADLINE.md`](V07_REAL_MODEL_HEADLINE.md) — real-model benchmark policy
- [`V073_PERFORMANCE_HARDENING.md`](V073_PERFORMANCE_HARDENING.md) — benchmark and runtime hardening

## Tokenizer hardening

v0.7.1 added hardened GGUF tokenization for GPT-2 byte-BPE and Llama/SentencePiece-style unigram tokenizers, including:

- arbitrary-byte fallback,
- malformed UTF-8 handling,
- special-token recognition,
- tokenizer CLI tooling,
- deterministic fixtures,
- fuzz tests, and
- SentencePiece reference-parity tests.

See [`V071_TOKENIZER_HARDENING.md`](V071_TOKENIZER_HARDENING.md) for the executed test matrix and remaining parity gates.

## Current limitations

MemVanta is still an experimental runtime. In particular:

- llama.cpp remains substantially faster in the published TinyStories CPU throughput comparison,
- broader architecture and tokenizer coverage is still evolving,
- large-model beyond-RAM behavior needs stronger trained-model benchmark evidence,
- shared GitHub runners can introduce measurable benchmark variance, and
- results from one CPU or virtualized host should not be generalized without rerunning the supplied benchmark protocol.

These limitations are intentionally documented rather than hidden behind headline numbers.

## Roadmap

Near-term benchmark and systems priorities are:

- publish the SmolLM2-360M same-model llama.cpp A/B result,
- extend trained-model validation to larger checkpoints,
- measure the memory/throughput curve as model size grows,
- test constrained-memory and beyond-RAM execution explicitly,
- improve decode throughput while preserving the low-memory design,
- broaden tokenizer/model-family compatibility, and
- continue reducing benchmark variance and improving reproducibility.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for build, test, and benchmark-evidence expectations.

For security issues, please follow [`SECURITY.md`](SECURITY.md).

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
