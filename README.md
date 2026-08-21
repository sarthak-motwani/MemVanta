# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![3B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/threeb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/threeb-model-ab.yml)
[![1.1B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/oneb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/oneb-model-ab.yml)
[![RAM Constrained 1B](https://github.com/sauravsingla/MemVanta/actions/workflows/ram-constrained.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ram-constrained.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Memory-adaptive C++ LLM inference for GGUF models.**

MemVanta is an experimental CPU-first runtime focused on **reducing resident-memory pressure** while executing quantized language models natively.

> **OpenLLaMA 3B v2:** **45.43% lower peak RSS** than pinned llama.cpp in a repeated exact-same-GGUF CPU benchmark. At 1.1B, MemVanta also completed under a **512 MiB cgroup memory ceiling** where llama.cpp was OOM-killed.

## 3B results

**OpenLLaMA 3B v2 Q4_0 · generated from a pinned official source revision · exact same GGUF · CPU only · 4 threads · pp512/tg128 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp | Comparison |
|---|---:|---:|---:|
| Prompt processing | 5.28 ± 0.03 tok/s | 24.46 ± 0.02 tok/s | llama.cpp 4.63× faster |
| Token generation | 3.43 ± 0.00 tok/s | 15.67 ± 0.04 tok/s | llama.cpp 4.57× faster |
| Peak RSS | **2070.52 MiB** | 3794.18 MiB | **MemVanta 45.43% lower** |

MemVanta used about **1.68 GiB less peak resident memory** in this run. This is a **memory-efficiency result, not a throughput win**.

The Q4_0 GGUF was generated in CI from the pinned official OpenLLaMA 3B v2 source using the pinned llama.cpp converter/quantizer, then fingerprinted before both runtimes used the exact same file.

Raw evidence: [`results/openllama-3b-v2-ab/`](results/openllama-3b-v2-ab/)

## 1.1B constrained-memory result

The repeated TinyLlama 1.1B same-GGUF A/B measured **45.88% lower peak RSS** for MemVanta (628.99 MiB vs 1162.23 MiB). A separate Linux cgroup-v2 pressure sweep then tested execution with swap disabled.

| Memory ceiling | MemVanta | llama.cpp |
|---:|:---:|:---:|
| 640 MiB | ✅ | ✅ |
| 576 MiB | ✅ | ❌ OOM |
| 512 MiB | ✅ | ❌ OOM |

Within the tested sweep, MemVanta's lowest successful ceiling was **512 MiB** versus **640 MiB** for llama.cpp — a **128 MiB / 20% lower successful memory ceiling**. This is an execution-under-pressure result, not an exact minimum physical-RAM requirement.

Raw evidence: [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/) · [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/)

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

Benchmark claims should be reproducible. Trained-model A/B workflows align the exact GGUF, CPU-only execution, thread count, context, prompt/generation workload, KV format, batch size, warm-up, repetitions, and pinned llama.cpp revision.

For the 3B benchmark, the workflow additionally pins the official source revision, generates the GGUF with the pinned llama.cpp conversion toolchain, and records the resulting SHA-256 before benchmarking.

Successful runs publish raw evidence into the repository rather than leaving it only in transient CI logs.

### Evidence

- [`results/openllama-3b-v2-ab/`](results/openllama-3b-v2-ab/) — OpenLLaMA 3B v2 repeated exact-same-GGUF A/B
- [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/) — TinyLlama 1.1B repeated same-model A/B
- [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/) — TinyLlama 1.1B cgroup memory-pressure sweep
- [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) — SmolLM2-360M repeated same-model A/B
- [`results/stories15m-repeated-ab/`](results/stories15m-repeated-ab/) — TinyStories 15M repeated same-model A/B
- [`V072_REAL_MODEL_AB.md`](V072_REAL_MODEL_AB.md) — A/B protocol
- [`V073_PERFORMANCE_HARDENING.md`](V073_PERFORMANCE_HARDENING.md) — runtime and benchmark hardening

## Status

MemVanta is an **engineering prototype under active development**. Published trained-model evidence now spans **15M → 3B parameters** and consistently shows lower peak RSS than the pinned llama.cpp comparison, while llama.cpp remains substantially faster on the published CPU workloads. A separate 1.1B cgroup-v2 sweep also demonstrates a lower tested execution ceiling under memory pressure.

Next milestones: **7B trained checkpoint**, physical-CPU reproduction, throughput optimization, finer constrained-memory boundary testing, and a direct real-model demonstration where the model exceeds available RAM.

## Contributing

Issues, benchmark reproductions, architecture support, kernel optimization, and pull requests are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

For security reports, see [`SECURITY.md`](SECURITY.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
