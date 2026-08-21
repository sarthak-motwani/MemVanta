# MemVanta CPU v0.7.2

[![Build and Test](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![Real Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml)
[![Medium Model Validation](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-validation.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-validation.yml)
[![Medium Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **Memory-adaptive LLM inference beyond conventional RAM and VRAM constraints.**

MemVanta is a CPU-first C++20 inference runtime for GGUF language models. It combines quantized execution, hierarchical memory management, compact KV caches, batched prefill, tokenizer support, and reproducible benchmark tooling in a systems-oriented runtime designed to explore how far useful LLM inference can be pushed under tight memory budgets.

The project is deliberately benchmark-driven: synthetic kernel results, trained-model validation, and direct same-model llama.cpp comparisons are reported separately so claims remain auditable.

## Headline result

On a trained **SmolLM2-360M Q4_0** checkpoint, using the exact same GGUF, CPU-only execution, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, one warm-up, and five measured repetitions:

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 46.62 ± 0.92 tok/s | 186.98 ± 0.77 tok/s | llama.cpp 4.01× faster |
| Token generation | 24.83 ± 0.07 tok/s | 110.88 ± 0.22 tok/s | llama.cpp 4.47× faster |
| Peak RSS | **266,700 KiB (260.45 MiB)** | 448,936 KiB (438.41 MiB) | **MemVanta 40.59% lower** |

**The current result is a memory-efficiency result, not a throughput win.** MemVanta used about **178 MiB less peak resident memory** in this 360M-parameter same-model run while llama.cpp remained substantially faster.

The benchmark is preserved under [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) with raw per-repetition data, environment information, process RSS evidence, llama.cpp JSON output, and workflow provenance.

## Why MemVanta

Most inference engines optimize primarily for throughput when the model comfortably fits in available memory. MemVanta explores a different systems question:

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

## Verified trained-model results

### 1. SmolLM2-360M Q4_0 — repeated MemVanta vs llama.cpp A/B

This is the strongest current same-model comparison.

**Protocol**

- Model: `SmolLM2-360M.Q4_0.gguf`
- Parameters: 361.82M
- Same GGUF and SHA-256 for both runtimes
- CPU only
- 4 threads
- pp512 / tg128
- Context: 768
- Batch / ubatch: 32
- KV: F16
- Warm-up: 1
- Measured repetitions: 5
- llama.cpp pinned to commit `6503355df0eb4f65875012523263c302fe0088c1`

| Metric | MemVanta | llama.cpp |
|---|---:|---:|
| Prompt processing mean | **46.62 tok/s** | **186.98 tok/s** |
| Prompt processing SD | 0.92 | 0.77 |
| Token generation mean | **24.83 tok/s** | **110.88 tok/s** |
| Token generation SD | 0.07 | 0.22 |
| Peak RSS | **266,700 KiB** | **448,936 KiB** |
| RSS difference | **40.59% lower** | — |

MemVanta's five prompt-processing samples ranged from **45.03 to 47.19 tok/s**, and generation ranged from **24.74 to 24.90 tok/s**, indicating low variance in this run.

Raw evidence: [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/)

### 2. TinyStories 15M Q4_0 — repeated MemVanta vs llama.cpp A/B

The smaller trained-model A/B independently shows the same direction on resident memory.

**Protocol:** same trained GGUF, CPU-only, 4 threads, context 128, pp64/tg64, batch 32, F16 KV, one warm-up and five measured repetitions.

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 2,107.44 ± 43.81 tok/s | 8,211.27 ± 428.90 tok/s | llama.cpp 3.90× faster |
| Token generation | 730.65 ± 76.04 tok/s | 2,007.99 ± 115.31 tok/s | llama.cpp 2.75× faster |
| Peak RSS | **30,676 KiB** | 43,204 KiB | **MemVanta 29.0% lower** |

MemVanta prompt-processing CV was 2.08%; generation CV was 10.41%, so the generation result carries a high-variance warning.

Raw evidence: [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/)

### 3. SmolLM2-360M native validation

Before the repeated A/B run, the medium-model validation gate independently demonstrated native trained-model execution with tokenizer and deterministic greedy-generation checks.

| Metric | Result |
|---|---:|
| Short-run prompt processing | **59.04 tok/s** |
| Short-run token generation | **34.42 tok/s** |
| Peak RSS | **244.32 MiB** |
| Model load time | **22.70 ms** |
| KV allocation | **5.00 MiB** |

This short validation uses a different workload from the pp512/tg128 A/B benchmark and therefore should not be compared numerically with the headline table.

Raw evidence: [`results/smollm2-360m/`](results/smollm2-360m/)

## What the current evidence says

Across two trained-model same-GGUF comparisons, MemVanta has shown lower peak RSS than the pinned llama.cpp reference:

| Model | MemVanta peak RSS | llama.cpp peak RSS | MemVanta reduction |
|---|---:|---:|---:|
| TinyStories 15M Q4_0 | 30,676 KiB | 43,204 KiB | **29.0%** |
| SmolLM2-360M Q4_0 | 266,700 KiB | 448,936 KiB | **40.59%** |

This is encouraging evidence that the low-memory design persists beyond a tiny diagnostic checkpoint. It is **not yet evidence that the advantage will scale monotonically to 1B, 3B, 7B, or beyond-RAM workloads**; those measurements remain future work.

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
  --threads 4 \
  --batch 32 \
  --kv f16 \
  --ctx 768 \
  --prompt 512 \
  --gen 128 \
  --reps 5 \
  --warmup 1
```

For benchmark claims, use repeated measurements rather than a single run and record the exact model SHA, CPU allocation, KV type, batch size, context, compiler, and MemVanta commit.

## Auto-tune

MemVanta can search thread and batch combinations instead of assuming maximum parallelism is optimal on every CPU:

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

A valid same-model A/B comparison keeps constant:

1. the exact GGUF file and SHA-256,
2. quantization format,
3. CPU-only/GPU-offload policy,
4. logical CPU allocation and thread count,
5. context length,
6. prompt and generation workload,
7. KV format,
8. batch and micro-batch configuration, and
9. warm-up and repetition count.

Published workflows preserve process RSS, environment details, exact runtime commits, raw outputs, and repeated throughput samples.

### Real Model A/B

The TinyStories workflow builds MemVanta and a pinned llama.cpp reference, verifies the same trained GGUF, runs repeated CPU-only A/B measurements, captures peak RSS and raw statistics, performs additional generation/profiling checks, and publishes successful evidence under `results/`.

### Medium Model Validation

The medium-model validation workflow checks SmolLM2-360M tokenizer behavior, deterministic greedy generation, native execution, timing, and peak RSS.

### Medium Model A/B

The medium-model A/B workflow runs the same trained SmolLM2-360M Q4_0 file through MemVanta and pinned llama.cpp with the declared pp512/tg128 protocol. Successful run `32442549603` is preserved in the repository.

## Evidence and reproducibility

Key evidence is committed rather than left only in transient CI logs:

- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — repeated SmolLM2-360M MemVanta/llama.cpp A/B
- [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/) — repeated TinyStories MemVanta/llama.cpp A/B
- [`results/smollm2-360m/`](results/smollm2-360m/) — SmolLM2-360M trained-model validation
- [`results/VALIDATION_GATES_2026-08-21.md`](results/VALIDATION_GATES_2026-08-21.md) — validation boundary
- [`V072_REAL_MODEL_AB.md`](V072_REAL_MODEL_AB.md) — repeated A/B protocol
- [`V07_REAL_MODEL_HEADLINE.md`](V07_REAL_MODEL_HEADLINE.md) — real-model benchmark policy
- [`V073_PERFORMANCE_HARDENING.md`](V073_PERFORMANCE_HARDENING.md) — benchmark and runtime hardening

## Tokenizer hardening

v0.7.1 added hardened GGUF tokenization for GPT-2 byte-BPE and Llama/SentencePiece-style unigram tokenizers, including arbitrary-byte fallback, malformed UTF-8 handling, special-token recognition, tokenizer CLI tooling, deterministic fixtures, fuzz tests, and SentencePiece reference-parity tests.

See [`V071_TOKENIZER_HARDENING.md`](V071_TOKENIZER_HARDENING.md) for the executed test matrix and remaining parity gates.

## Current limitations

MemVanta is still an experimental runtime. In particular:

- llama.cpp remains substantially faster in both published trained-model CPU throughput comparisons,
- the 360M A/B result comes from a shared virtualized GitHub runner and should be reproduced on dedicated physical hardware,
- broader architecture and tokenizer coverage is still evolving,
- trained-model evidence currently stops at the 360M-parameter class,
- constrained-memory and true beyond-RAM behavior still need direct trained-model A/B evidence, and
- results from one CPU or virtualized host should not be generalized without rerunning the supplied protocol.

These limitations are intentionally documented rather than hidden behind headline numbers.

## Roadmap

Near-term benchmark and systems priorities are:

- reproduce the 360M A/B result on dedicated physical CPU hardware,
- extend trained-model validation to approximately 1B, 3B, and 7B checkpoints,
- measure the memory/throughput curve as model size grows,
- test constrained-memory and beyond-RAM execution explicitly,
- improve prefill and decode throughput while preserving the low-memory design,
- isolate where the current llama.cpp throughput gap originates,
- broaden tokenizer/model-family compatibility, and
- continue reducing benchmark variance and improving reproducibility.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for build, test, and benchmark-evidence expectations.

For security issues, please follow [`SECURITY.md`](SECURITY.md).

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
