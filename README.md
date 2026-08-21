# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![1.1B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/oneb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/oneb-model-ab.yml)
[![RAM Constrained 1B](https://github.com/sauravsingla/MemVanta/actions/workflows/ram-constrained.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ram-constrained.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Memory-adaptive C++ LLM inference for GGUF models.**

MemVanta is an experimental CPU-first runtime focused on **reducing resident-memory pressure** while executing quantized language models natively.

> **TinyLlama 1.1B:** **45.88% lower peak RSS** than pinned llama.cpp in a repeated same-GGUF benchmark, and successful execution at a **512 MiB cgroup memory ceiling** where llama.cpp was OOM-killed.

## 1.1B results

### Same-model A/B

**TinyLlama 1.1B Q4_0 · same GGUF · CPU only · 4 threads · pp512/tg128 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 17.12 ± 0.04 tok/s | 79.00 ± 0.28 tok/s | llama.cpp 4.61× faster |
| Token generation | 5.79 ± 0.00 tok/s | 48.58 ± 0.06 tok/s | llama.cpp 8.39× faster |
| Peak RSS | **628.99 MiB** | 1162.23 MiB | **MemVanta 45.88% lower** |

MemVanta saved about **533 MiB of peak resident memory** in this run. This is a **memory-efficiency result, not a throughput win**.

Raw evidence: [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/)

### Constrained-memory boundary

**Linux cgroup-v2 `MemoryMax` · swap disabled · same TinyLlama 1.1B Q4_0 GGUF · 4 threads · pp128/tg32 · context 768 · batch 32 · F16 KV**

| Memory ceiling | MemVanta | llama.cpp |
|---:|:---:|:---:|
| 640 MiB | ✅ | ✅ |
| 576 MiB | ✅ | ❌ OOM |
| 512 MiB | ✅ | ❌ OOM |

Within the tested sweep, MemVanta's lowest successful ceiling was **512 MiB** versus **640 MiB** for llama.cpp — a **128 MiB / 20% lower successful memory ceiling**. This is an execution-under-pressure result, not a throughput comparison.

Raw evidence: [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/)

Supporting trained-model A/B results show the same memory direction:
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
- reproducible MemVanta vs llama.cpp A/B and memory-pressure benchmarks

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
- [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/) — TinyLlama 1.1B cgroup memory-pressure sweep
- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — SmolLM2-360M repeated same-model A/B
- [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/) — TinyStories 15M repeated same-model A/B
- [`V072_REAL_MODEL_AB.md`](V072_REAL_MODEL_AB.md) — A/B protocol
- [`V073_PERFORMANCE_HARDENING.md`](V073_PERFORMANCE_HARDENING.md) — runtime and benchmark hardening

## Status

MemVanta is an **engineering prototype under active development**. Current evidence shows a consistent memory-efficiency direction from 15M to 1.1B parameters and a verified constrained-memory advantage at 1.1B, while llama.cpp remains substantially faster on the published CPU workloads.

Next milestones: **3B → 7B trained checkpoints**, physical-CPU reproduction, throughput optimization, and a direct real-model demonstration where the model exceeds available RAM.

## Contributing

Issues, benchmark reproductions, architecture support, kernel optimization, and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

For security reports, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
