# MemVanta CPU v0.3 benchmark report

Date: 2026-08-20

## Benchmark policy

MemVanta separates **microbenchmarks** from **end-to-end LLM benchmarks**. This report follows the statistical conventions used by llama.cpp's `llama-bench` for CPU inference work: warm-up before timing, repeated runs, mean throughput and standard deviation, fixed shapes/thread counts, and machine/compiler disclosure. We use 10 measured repetitions per kernel configuration.

For a complete LLM engine, the project will additionally report the user-facing metrics used by MLPerf Client: **time-to-first-token (TTFT)** and **output tokens/second (TPS)**. MemVanta v0.3 does not yet claim those metrics because the full GGUF transformer/tokenizer graph is not implemented. No synthetic token/s is presented as if it were real model inference.

## Test host

- CPU: AMD EPYC 9V74 (virtualized)
- 5 vCPUs available
- ISA exposed: AVX2/FMA plus AVX-512 family
- RAM: ~5.9 GiB, no swap
- Linux x86-64
- GCC 14.2.0
- C++20, Release, `-O3 -march=native`

## v0.3 change: vectorized Q4_0

The Q4_0 dot product now uses AVX2 to unpack 4-bit nibbles in vector registers, sign-center them to [-8, 7], convert to float vectors and FMA against the activation vector. This removes the scalar nibble-unpack bottleneck measured in v0.2.

## Kernel benchmark

10 measured repetitions after one warm-up invocation per configuration.

| Kernel | Shape | Threads | Mean | SD | GFLOP/s | Weight GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| F32 | 2048 x 2048 | 1 | 3.430 ms | 0.068 ms | 2.445 | 4.555 |
| Q8_0 | 2048 x 2048 | 1 | 0.428 ms | 0.008 ms | 19.584 | 10.259 |
| Q4_0 | 2048 x 2048 | 1 | 0.682 ms | 0.098 ms | 12.301 | 3.580 |
| F32 | 2048 x 2048 | 5 | 1.118 ms | 0.309 ms | 7.505 | 13.980 |
| Q8_0 | 2048 x 2048 | 5 | 0.280 ms | 0.029 ms | 29.953 | 15.691 |
| Q4_0 | 2048 x 2048 | 5 | 0.365 ms | 0.073 ms | 23.011 | 6.697 |
| F32 | 4096 x 4096 | 1 | 15.001 ms | 2.683 ms | 2.237 | 4.166 |
| Q8_0 | 4096 x 4096 | 1 | 1.861 ms | 0.084 ms | 18.033 | 9.447 |
| Q4_0 | 4096 x 4096 | 1 | 2.624 ms | 0.040 ms | 12.790 | 3.722 |
| F32 | 4096 x 4096 | 5 | 3.711 ms | 0.871 ms | 9.041 | 16.841 |
| Q8_0 | 4096 x 4096 | 5 | 0.769 ms | 0.072 ms | 43.633 | 22.858 |
| Q4_0 | 4096 x 4096 | 5 | 1.134 ms | 0.146 ms | 29.576 | 8.608 |
| F32 | 11008 x 4096 | 1 | 38.419 ms | 0.454 ms | 2.347 | 4.372 |
| Q8_0 | 11008 x 4096 | 1 | 5.262 ms | 0.153 ms | 17.139 | 8.979 |
| Q4_0 | 11008 x 4096 | 1 | 7.185 ms | 0.149 ms | 12.552 | 3.653 |
| F32 | 11008 x 4096 | 5 | 13.830 ms | 3.512 ms | 6.520 | 12.145 |
| Q8_0 | 11008 x 4096 | 5 | 2.503 ms | 0.420 ms | 36.028 | 18.874 |
| Q4_0 | 11008 x 4096 | 5 | 3.047 ms | 0.419 ms | 29.599 | 8.615 |

### Q4_0 improvement over v0.2

On the directly comparable 4096 x 4096 / 5-thread case, Q4_0 rose from 8.228 to 29.576 GFLOP/s: **3.59x**. On 11008 x 4096 / 5-thread, it rose from 10.089 to 29.599 GFLOP/s: **2.93x**.

## Thread scaling: 4096 x 4096

| Threads | Q8_0 GFLOP/s | Q4_0 GFLOP/s |
|---:|---:|---:|
| 1 | 18.765 | 12.608 |
| 2 | 21.605 | 18.068 |
| 3 | 25.151 | 26.559 |
| 4 | 27.812 | 25.290 |
| 5 | 37.384 | 30.210 |

The virtualized host shows noise/non-monotonicity around 3-4 threads, so these measurements should not be generalized to physical desktop/server CPUs without rerunning the supplied benchmark script.

## Memory-capacity result retained from v0.2

The memory engine successfully traversed an 8 GiB sparse model-sized mapping on a machine with ~5.9 GiB physical RAM and no swap with ~67 MiB peak process RSS. This is a **capacity test only**, not an NVMe-bandwidth claim.

## What qualifies as an industry-standard end-to-end comparison next

The next release must run the exact same GGUF model and quantization through MemVanta and llama.cpp and report at least:

- prompt processing: `pp512`
- generation: `tg128`
- TTFT
- output TPS excluding first token
- peak RSS / model-load RSS
- model load time
- fixed CPU thread count and affinity
- fixed context/KV type/batch settings
- at least 5-10 repeated samples with mean and variation
- exact model SHA256, compiler, flags and commit IDs
- quality sanity check (perplexity or fixed-token logits tolerance)

Until that graph exists, this repository deliberately reports kernel/memory performance instead of inventing end-to-end token/s.
