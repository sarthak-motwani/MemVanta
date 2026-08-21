# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![Real Model A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/real-model-ab.yml)
[![SmolLM2-360M A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Memory-adaptive C++ LLM inference for GGUF models.**

MemVanta is an experimental CPU-first inference runtime focused on **reducing resident memory pressure** while executing quantized language models natively. It combines direct GGUF loading, quantized kernels, paged KV cache, mmap-backed tensor access, batched prefill, tokenizer support, and reproducible same-model benchmarks.

> **Latest trained-model result:** on SmolLM2-360M Q4_0, MemVanta used **40.59% less peak RSS** than pinned llama.cpp in a repeated CPU-only same-GGUF benchmark.

## Latest benchmark

**SmolLM2-360M Q4_0 · same GGUF · CPU only · 4 threads · pp512/tg128 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 46.62 ± 0.92 tok/s | 186.98 ± 0.77 tok/s | llama.cpp 4.01× faster |
| Token generation | 24.83 ± 0.07 tok/s | 110.88 ± 0.22 tok/s | llama.cpp 4.47× faster |
| Peak RSS | **260.45 MiB** | 438.41 MiB | **MemVanta 40.59% lower** |

MemVanta saved about **178 MiB of peak resident memory** in this run. This is a **memory-efficiency result, not a throughput win**.

Raw evidence: [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/)

A second trained-model A/B on TinyStories 15M showed the same direction: **29.0% lower peak RSS** than the same pinned llama.cpp reference. See [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/).

## Why MemVanta?

Modern inference engines are exceptionally optimized for throughput. MemVanta explores a complementary systems problem:

**What can an LLM runtime do when memory—not just compute—is the constraint?**

The project currently focuses on:

- lower resident-memory pressure
- native GGUF execution
- Q4_0 / Q8_0 / F16 / F32 paths
- paged F32 / F16 / Q8 KV cache
- mmap-backed tensor access and bounded caching
- AVX2/FMA quantized kernels
- batched prefill and dedicated decode paths
- GPT-2 and Llama/SentencePiece-style tokenization
- deterministic trained-model validation
- reproducible MemVanta vs llama.cpp A/B benchmarks

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run a benchmark

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

## Benchmark philosophy

Benchmark claims should be reproducible. MemVanta's trained-model A/B workflows keep the important variables aligned: exact GGUF and SHA-256, CPU-only execution, thread count, context, prompt/generation workload, KV format, batch size, warm-up, repetitions, and pinned llama.cpp revision.

Successful CI runs publish the raw evidence into the repository rather than leaving it only in transient logs.

### Evidence

- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — SmolLM2-360M repeated same-model A/B
- [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/) — TinyStories 15M repeated same-model A/B
- [`results/smollm2-360m/`](results/smollm2-360m/) — native SmolLM2-360M validation
- [`V072_REAL_MODEL_AB.md`](V072_REAL_MODEL_AB.md) — A/B protocol
- [`V073_PERFORMANCE_HARDENING.md`](V073_PERFORMANCE_HARDENING.md) — runtime and benchmark hardening

## Status

MemVanta is an **engineering prototype under active development**. The current evidence supports a promising memory-efficiency direction, but llama.cpp remains substantially faster on the published CPU workloads.

The next milestones are larger trained checkpoints (≈1B → 3B → 7B), dedicated physical-CPU reproduction, explicit constrained-memory testing, and a direct demonstration of inference when the model exceeds available RAM.

## Contributing

Issues, benchmark reproductions, architecture support, kernel optimization, and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

For security reports, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
