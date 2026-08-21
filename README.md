# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![1.1B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/oneb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/oneb-model-ab.yml)
[![SmolLM2-360M A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/medium-model-ab.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Memory-adaptive C++ LLM inference for GGUF models.**

MemVanta is an experimental CPU-first runtime focused on **reducing resident-memory pressure** while executing quantized language models natively.

> **Verified 1.1B result:** on TinyLlama 1.1B Q4_0, MemVanta used **45.88% less peak RSS** than pinned llama.cpp in a repeated CPU-only same-GGUF benchmark.

## Latest benchmark

**TinyLlama 1.1B Q4_0 · same GGUF · CPU only · 4 threads · pp512/tg128 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 17.12 ± 0.04 tok/s | 79.00 ± 0.28 tok/s | llama.cpp 4.61× faster |
| Token generation | 5.79 ± 0.00 tok/s | 48.58 ± 0.06 tok/s | llama.cpp 8.39× faster |
| Peak RSS | **628.99 MiB** | 1162.23 MiB | **MemVanta 45.88% lower** |

MemVanta saved about **533 MiB of peak resident memory** in this run. This is a **memory-efficiency result, not a throughput win**.

Raw evidence: [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/)

Supporting trained-model results show the same direction:
- **SmolLM2-360M:** 40.59% lower peak RSS — [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/)
- **TinyStories 15M:** 29.0% lower peak RSS — [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/)

## Why MemVanta?

Modern inference engines are exceptionally optimized for throughput. MemVanta explores a complementary systems question:

**What can an LLM runtime do when memory—not just compute—is the constraint?**

Current focus:

- lower resident-memory pressure
- native GGUF execution
- Q4_0 / Q6_K / Q8_0 / F16 / F32 execution paths
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

Benchmark claims should be reproducible. Trained-model A/B workflows align the exact GGUF and SHA-256, CPU-only execution, thread count, context, prompt/generation workload, KV format, batch size, warm-up, repetitions, and pinned llama.cpp revision.

Successful runs publish raw evidence into the repository rather than leaving it only in transient CI logs.

### Evidence

- [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/) — TinyLlama 1.1B repeated same-model A/B
- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — SmolLM2-360M repeated same-model A/B
- [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/) — TinyStories 15M repeated same-model A/B
- [`results/smollm2-360m/`](results/smollm2-360m/) — native SmolLM2-360M validation
- [`V072_REAL_MODEL_AB.md`](V072_REAL_MODEL_AB.md) — A/B protocol
- [`V073_PERFORMANCE_HARDENING.md`](V073_PERFORMANCE_HARDENING.md) — runtime and benchmark hardening

## Status

MemVanta is an **engineering prototype under active development**. Current evidence supports a consistent memory-efficiency direction from 15M to 1.1B parameters, while llama.cpp remains substantially faster on the published CPU workloads.

Next milestones: **RAM-constrained 1.1B testing**, larger trained checkpoints (3B → 7B), physical-CPU reproduction, and a direct demonstration of inference when the model exceeds available RAM.

## Contributing

Issues, benchmark reproductions, architecture support, kernel optimization, and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

For security reports, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
