# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![3B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/threeb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/threeb-model-ab.yml)
[![RAM Constrained 1B](https://github.com/sauravsingla/MemVanta/actions/workflows/ram-constrained.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ram-constrained.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Memory-efficient CPU LLM inference for GGUF models.**

MemVanta is an experimental C++20 inference runtime for **quantized GGUF language models on CPUs**, designed for systems where **RAM is the constraint**. It explores mmap-backed model access, bounded caching, quantized kernels, paged KV cache, and reproducible memory-pressure benchmarking.

> **OpenLLaMA 3B v2 Q4_0:** MemVanta used **45.43% less peak RSS** than pinned llama.cpp on the exact same GGUF. llama.cpp remained ~4.6× faster, so this is a **memory-efficiency result, not a throughput win**.

## Who is this for?

MemVanta is currently most useful to:

- **LLM systems researchers** studying memory-efficient and CPU inference
- **C++ / inference-engine developers** working with GGUF, quantization, mmap, KV cache, and low-level kernels
- **Edge AI developers** targeting low-RAM CPUs, mini PCs, VMs, and constrained machines
- **Benchmarking and infrastructure engineers** comparing inference runtimes under fixed memory budgets
- **Students and contributors** interested in the internals of local LLM inference

It is **not yet a drop-in replacement for llama.cpp** or a polished end-user chatbot runtime.

## Verified result

**OpenLLaMA 3B v2 Q4_0 · same generated GGUF · CPU only · 4 threads · pp512/tg128 · batch 32 · F16 KV · 5 measured runs**

| Metric | MemVanta | llama.cpp |
|---|---:|---:|
| Peak RSS | **2070.52 MiB** | 3794.18 MiB |
| Prompt processing | 5.28 tok/s | 24.46 tok/s |
| Token generation | 3.43 tok/s | 15.67 tok/s |

**Peak resident-memory reduction: 45.43% (~1.68 GiB).**

Raw evidence: [`results/openllama-3b-v2-ab/`](results/openllama-3b-v2-ab/)

### Memory-pressure proof

In a separate TinyLlama 1.1B cgroup-v2 test with swap disabled, MemVanta completed at a **512 MiB tested memory ceiling**, while pinned llama.cpp was OOM-killed at 576 MiB and 512 MiB; llama.cpp's lowest successful tested ceiling was 640 MiB.

Raw evidence: [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/)

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

- [`results/openllama-3b-v2-ab/`](results/openllama-3b-v2-ab/) — 3B repeated same-GGUF A/B
- [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/) — constrained-memory sweep
- [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/) — 1.1B repeated A/B
- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — 360M repeated A/B

## Project direction

The goal is simple:

> **Run larger quantized LLMs within smaller RAM budgets on commodity CPUs.**

Current work is focused on **7B validation, tighter constrained-memory boundaries, physical-CPU reproduction, and throughput optimization**.

MemVanta is an **engineering prototype under active development**. Published results show a memory-efficiency advantage on the tested models; they do not establish a universal scaling law or throughput advantage.

## Contributing

Issues, benchmark reproductions, kernel optimizations, model compatibility work, and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

For security reports, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
