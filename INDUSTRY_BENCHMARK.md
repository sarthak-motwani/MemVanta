# MemVanta CPU v0.5 — Industry-aligned CPU benchmark

Date: 2026-08-20

## What changed in v0.5

The decoder proxy now matches the public SmolLM2-135M architecture: 576 hidden width, 1536 FFN, 30 layers, 9 query heads, 3 KV heads (GQA), 64-d head size, 49,152 vocabulary and RoPE theta 100,000. Its measured weight-bearing parameter count is 134.48M, closely matching the public 134.5M model. Q/K/V dimensions now implement GQA rather than full multi-head K/V.

## Industry-aligned protocol

- llama.cpp-style core tests: `pp512` and `tg128`
- MLPerf Client-style interactive workload: 128 input tokens / 256 output tokens
- TTFT reported separately from subsequent output tokens/s
- 5 independent measured runs, each preceded by a warm-up
- fixed 5 CPU threads; CPU affinity available as 0-4
- Release C++20 build, `-O3 -march=native`
- sample mean, sample standard deviation, median, P95, min/max and coefficient of variation
- OS peak RSS captured with `getrusage`/`time -v`

This is **metric/protocol aligned**, not an MLPerf submission and not yet an apples-to-apples llama.cpp competitor result. MemVanta still uses deterministic synthetic Q8_0 weights and sequential prefill.

## Host

- CPU: AMD EPYC 9V74 (KVM virtualized)
- CPUs exposed: 5 (0-4)
- RAM: ~5.9 GiB
- Swap: 0
- SIMD exposed: AVX2/FMA and AVX-512
- Compiler: GCC 14.2

## Architecture-matched workload

- decoder: Llama / SmolLM2-135M shape
- parameters: 134.48M
- packed Q8_0 weight footprint: 144.28 MiB
- KV capacity used by benchmark: 45.00 MiB
- GQA: 9 query heads / 3 KV heads
- max benchmark context: 1024

## Five-run results

| Metric | Mean ± SD | Median | P95 | Min | Max | CV |
|---|---:|---:|---:|---:|---:|---:|
| pp512 | 63.07 ± 11.05 tok/s | 70.12 | 72.33 | 50.17 | 72.84 | 17.5% |
| tg128 | 66.51 ± 12.46 tok/s | 66.41 | 80.58 | 48.71 | 83.94 | 18.7% |
| TTFT (128 input + first output) | 1152.54 ± 124.25 ms | 1198.21 | 1283.91 | 1022.00 | 1300.40 | 10.8% |
| output TPS (excl. first token) | 63.91 ± 6.42 tok/s | 63.84 | 71.55 | 56.58 | 72.58 | 10.0% |

Peak RSS across completed runs was approximately **194.05–194.27 MiB** with zero swap.

## Raw samples

| Run | pp512 tok/s | tg128 tok/s | TTFT ms | output tok/s |
|---:|---:|---:|---:|---:|
| 1 | 72.837 | 83.941 | 1024.15 | 72.576 |
| 2 | 50.175 | 67.127 | 1198.21 | 63.845 |
| 3 | 70.124 | 66.411 | 1300.40 | 56.575 |
| 4 | 51.911 | 66.345 | 1217.93 | 59.102 |
| 5 | 70.287 | 48.709 | 1022.00 | 67.463 |

## Interpretation

1. Generation is in the ~64–67 tok/s central range on this 5-vCPU VM, with a 66.41 tok/s median for `tg128`.
2. Median TTFT for the 128-token client workload is about 1.20 s.
3. Peak resident memory is below 200 MiB for a 134.48M-parameter Q8-shaped decoder plus KV cache.
4. Run-to-run variance is high (especially pp/tg) because this is a shared virtualized CPU host. These numbers are suitable for engineering regression tracking but should not be used for a public speed-vs-llama.cpp headline. A dedicated bare-metal runner is the next benchmarking requirement.
5. `pp512` is still sequential token prefill. llama.cpp normally batches prompt processing, so its `pp512` is not directly comparable yet.

## External benchmark gate

The first publishable competitor table requires: (a) direct GGUF loading, (b) exact same GGUF SHA256, (c) matching quantization and KV types, (d) batched prefill, (e) deterministic logit/token parity, (f) same thread/affinity and context settings, and (g) at least five warm measured repetitions.

Target public checkpoint for that gate: `QuantFactory/SmolLM2-135M-Instruct.Q4_K_M.gguf`, public SHA256 `8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684`.

## References used for protocol

- llama.cpp `llama-bench`: default prompt length 512 and generation length 128; reports average t/s with variability and machine/model settings.
- MLPerf Client: reports TTFT separately and tokens/s for subsequent output; content-generation category is approximately 128 input / 256 output tokens.

