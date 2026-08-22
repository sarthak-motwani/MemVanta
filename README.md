# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![7B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-model-ab.yml)
[![RAM Constrained 7B](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-ram-constrained.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-ram-constrained.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Memory-efficient CPU LLM inference for GGUF models.**

MemVanta is an experimental C++20 inference runtime for **quantized GGUF language models on CPUs**, designed for systems where **RAM is the constraint**. It explores mmap-backed model access, bounded caching, quantized kernels, paged KV cache, and reproducible memory-pressure benchmarking.

> **OpenLLaMA 7B v2 Q4_0:** MemVanta used **47.50% less peak RSS** than pinned llama.cpp on the exact same GGUF across 5 measured CPU runs. Under a separate cgroup-v2 memory sweep with swap disabled, MemVanta completed at a **3584 MiB tested ceiling** where llama.cpp was **OOM-killed**. llama.cpp remains substantially faster, so this is a **memory-efficiency result, not a throughput win**.

## Who is this for?

MemVanta is currently most useful to:

- **LLM systems researchers** studying memory-efficient and CPU inference
- **C++ / inference-engine developers** working with GGUF, quantization, mmap, KV cache, and low-level kernels
- **Edge AI developers** targeting low-RAM CPUs, mini PCs, VMs, and constrained machines
- **Benchmarking and infrastructure engineers** comparing inference runtimes under fixed memory budgets
- **Students and contributors** interested in the internals of local LLM inference

It is **not yet a drop-in replacement for llama.cpp** or a polished end-user chatbot runtime.

## Verified 7B result

**OpenLLaMA 7B v2 Q4_0 · exact same GGUF · CPU only · 4 threads · pp512/tg128 · context 768 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp |
|---|---:|---:|
| Peak RSS | **3983412 KiB (~3.80 GiB)** | 7586960 KiB (~7.24 GiB) |
| Prompt processing | 3.15 ± 0.04 tok/s | 45.82 ± 0.96 tok/s |
| Token generation | 1.52 ± 0.02 tok/s | 7.76 ± 0.09 tok/s |

**Peak resident-memory reduction: 47.50% (~3.44 GiB less).**

Raw evidence: [`results/openllama-7b-v2-ab/`](results/openllama-7b-v2-ab/)

### 7B memory-pressure proof

In a separate Linux cgroup-v2 `MemoryMax` sweep with swap disabled, using the **exact same OpenLLaMA 7B v2 Q4_0 GGUF**, CPU-only, 4 threads, pp128/tg32, context 768, batch 32, and F16 KV:

- MemVanta completed at a **3584 MiB tested memory ceiling**.
- pinned llama.cpp was **OOM-killed at 3584 MiB**.
- llama.cpp's lowest successful **tested** ceiling was **3840 MiB**.
- tested-ceiling difference: **256 MiB (6.67%)**.

This is an execution-under-pressure result over the tested sweep, **not an exact minimum physical-RAM requirement**.

Raw evidence: [`results/openllama-7b-v2-ram-constrained/`](results/openllama-7b-v2-ram-constrained/)

## What MemVanta implements

- native GGUF model execution
- Q4_0 / Q6_K / Q8_0 / F16 / F32 paths
- mmap-backed tensor access and bounded caching
- paged F32 / F16 / Q8 KV cache
- AVX2/FMA quantized kernels
- batched prefill and decode paths
- GPT-2 and Llama/SentencePiece-style tokenization
- CPU-only trained-model benchmarking against pinned llama.cpp
- reproducible constrained-memory experiments

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Benchmark a GGUF model

```bash
./build/memvanta_real_bench \
  --model model.gguf \
  --threads 4 \
  --ctx 768 \
  --batch 32 \
  --kv f16 \
  --prompt 512 \
  --gen 128 \
  --reps 5 \
  --warmup 1
```

## Benchmark evidence

Published runs preserve raw evidence instead of relying only on transient CI logs.

- [`results/openllama-7b-v2-ab/`](results/openllama-7b-v2-ab/) — 7B repeated exact-same-GGUF A/B
- [`results/openllama-7b-v2-ram-constrained/`](results/openllama-7b-v2-ram-constrained/) — 7B constrained-memory sweep
- [`results/openllama-3b-v2-ab/`](results/openllama-3b-v2-ab/) — 3B repeated same-GGUF A/B
- [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/) — 1.1B constrained-memory sweep
- [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/) — 1.1B repeated A/B
- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — 360M repeated A/B

### Validation protocol

MemVanta treats memory efficiency as the primary optimization target and throughput as a secondary transparency metric.

- [`docs/MEMORY_BENCHMARKING.md`](docs/MEMORY_BENCHMARKING.md) — benchmark methodology and claim boundaries
- [`docs/BENCHMARK_CHECKLIST.md`](docs/BENCHMARK_CHECKLIST.md) — publication checklist for new results
- [`docs/EXTERNAL_REPRODUCTION.md`](docs/EXTERNAL_REPRODUCTION.md) — guide for independent third-party reproduction

Independent results that confirm, narrow, or contradict the published measurements are welcome.

## Project direction

The goal is simple:

> **Run larger quantized LLMs within smaller RAM budgets on commodity CPUs.**

Current work is focused on **tightening the 7B/8B memory boundary, reproducing results on physical CPUs, broadening model-family coverage, and strengthening numerical and third-party validation**. Throughput remains reported so the trade-off is visible, but it is not the primary optimization target.

MemVanta is an **engineering prototype under active development**. Published trained-model evidence now reaches 7B and consistently shows lower peak RSS on the tested workloads; it does not establish a universal scaling law or throughput advantage.

## Contributing

Issues, benchmark reproductions, kernel optimizations, model compatibility work, and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

For security reports, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
