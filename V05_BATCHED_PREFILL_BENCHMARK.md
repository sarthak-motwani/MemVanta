# MemVanta CPU v0.5 — Batched Prefill + Compact KV Cache

Date: 2026-08-20

## v0.5 implementation

v0.5 adds the CPU features targeted after the v0.4 full GGUF decoder:

- batched causal prefill over multiple prompt tokens
- Q4_0/Q8_0 batched matrix execution (`tensor_matmul_batch`)
- a persistent C++ worker pool used by batched kernels and attention
- AVX2/FMA FP32 attention dot-product and accumulation primitives
- paged KV cache selectable as FP32, FP16, or per-token symmetric Q8
- a dedicated single-token generation path so decode does not pay batched-buffer/worker-pool overhead
- `--batch` and `--kv f32|f16|q8` benchmark controls
- correctness tests for the worker pool and FP32/FP16/Q8 KV paths

The project builds with C++20 and CTest passes.

## Benchmark protocol

The primary systems test follows llama.cpp `llama-bench` headline defaults where applicable:

- prompt processing: **pp512**
- token generation: **tg128**
- **5 measured runs**
- fixed 5 CPU threads
- CPU only
- same GGUF fixture and quantization for v0.4 and v0.5
- mean and sample standard deviation reported
- no run discarded

The fixture is the deterministic SmolLM2-135M-shape GGUF used for v0.4. It has SmolLM2-scale dimensions and Q4_0/Q8_0 tensor types, but **synthetic weights**. This is a systems-throughput benchmark, not a trained-model quality benchmark and not yet a llama.cpp-vs-MemVanta result.

### Host

- AMD EPYC 9V74 (virtualized)
- 5 allocated CPU cores
- ~5.9 GiB RAM
- GCC 14 / C++20
- Release, `-O3 -march=native`
- Linux x86-64

## v0.5 primary result (FP32 KV, batch=64)

| Run | pp512 tok/s | tg128 tok/s |
|---:|---:|---:|
| 1 | 154.72 | 69.55 |
| 2 | 153.08 | 74.34 |
| 3 | 115.51 | 67.19 |
| 4 | 119.62 | 62.46 |
| 5 | 125.09 | 64.30 |
| **mean ± sample SD** | **133.60 ± 18.85** | **67.57 ± 4.66** |
| **median** | **125.09** | **67.19** |

Peak RSS was about **124 MiB**. The final benchmark stage had **11.25 MiB** of allocated FP32 KV pages.

## v0.4 → v0.5 same-fixture comparison

The retained v0.4 five-run result was:

| Version | pp512 | tg128 |
|---|---:|---:|
| v0.4 | 49.64 ± 12.62 tok/s | 52.69 ± 8.18 tok/s |
| **v0.5** | **133.60 ± 18.85 tok/s** | **67.57 ± 4.66 tok/s** |
| **ratio** | **2.69×** | **1.28×** |

So the principal v0.5 target — prompt processing — improved by about **169%** on this fixture. Decode also improved by about **28%**, largely because single-token decode keeps the direct path while reusing vectorized attention.

These are engineering measurements on a shared virtualized host, not controlled bare-metal submissions.

## Batch-size sweep

A one-run sweep was used only for tuning insight, not as the primary 5-run claim:

| Batch | pp512 tok/s |
|---:|---:|
| 1 | 42.33 |
| 8 | 90.94 |
| 16 | 113.99 |
| 32 | 156.59 |
| 64 | 127.46 |
| 128 | 112.28 |

The local optimum in that sweep was batch 32, showing that larger batches are not automatically better on a 5-core CPU. The 5-run primary benchmark uses batch 64 to keep the release protocol fixed; future auto-tuning should choose batch size from measured hardware behavior.

## KV-cache modes

A controlled pp128/tg32 smoke workload on the same fixture demonstrates the memory trade-off:

| KV type | KV pages allocated | Peak RSS | pp128 | tg32 |
|---|---:|---:|---:|---:|
| FP32 | 11.25 MiB | 111.09 MiB | 116.34 tok/s | 41.11 tok/s |
| FP16 | 5.62 MiB | 105.46 MiB | 92.42 tok/s* | 38.71 tok/s* |
| Q8 | 2.87 MiB | 102.77 MiB | 111.45 tok/s | 45.04 tok/s |

`*` FP16 numbers shown were measured before the dedicated decode fast path was restored, so they should not be used for throughput ranking. The memory figure is valid. Q8 cuts allocated KV-page bytes by roughly **74.5%** versus FP32 for this workload; FP16 cuts them by **50%**.

The Q8 cache uses a symmetric per-token scale for K and V. Unit tests bound its dot/accumulation error against FP32; this is an engineering cache format, not claimed to match llama.cpp's exact Q8 KV representation.

## What is still required for a publishable llama.cpp comparison

The exact public trained GGUF must be present locally and the same file must be used by both engines. The runtime still cannot retrieve the Hugging Face Xet/LFS binary in this environment, so no MemVanta-vs-llama.cpp speed claim is fabricated.

Once the model is available, the apples-to-apples run should use:

- same GGUF SHA256
- same CPU/thread affinity
- CPU-only backend
- same KV type (llama-bench currently defaults K/V to F16)
- pp512 / tg128
- 5 repetitions with warm-up
- identical batch/ubatch policy where possible
- report mean ± SD, peak RSS, load time, TTFT, and output TPS

## Reproduce v0.5

```bash
cmake -S . -B build_v05 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v05 -j
ctest --test-dir build_v05 --output-on-failure

./build_v05/memvanta_real_bench \
  --model /path/to/model.gguf \
  --threads 5 --ctx 1024 \
  --prompt 512 --gen 128 \
  --batch 64 --kv f32 \
  --reps 5 --warmup 1
```

Compact-cache variants:

```bash
# half-size KV pages
... --kv f16

# ~quarter-size KV pages plus per-token scales
... --kv q8
```

## Next optimization targets

1. native register-blocked Q4/Q8 GEMM rather than repeated quantized GEMV across batch rows
2. pack/quantize activations once per batch and use integer dot-product paths
3. remove temporary `std::vector` copies in batched RoPE
4. SIMD FP16/Q8 KV attention instead of scalar dequantization loops
5. hardware auto-tuning for prompt batch size and worker scheduling
6. exact public-model parity and `llama-bench` A/B once the same trained GGUF is locally available
