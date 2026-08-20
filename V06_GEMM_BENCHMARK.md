# MemVanta CPU v0.6 — Register-blocked GEMM + SIMD KV + Auto-tuning

## What changed

v0.6 targets prompt-prefill throughput while keeping decode stable:

- 4-column register-blocked Q4_0 and Q8_0 GEMM microkernels
- one quantized weight decode reused across four activations
- optional batch-wide Q8 activation quantization (`MEMVANTA_V06_Q8_ACT=1`)
- persistent worker pool retained for batched prefill
- dedicated decode fast path retained (single-token path does not pay batch worker-pool overhead)
- SIMD F16 KV dot/accumulate with AVX2/F16C
- SIMD Q8 KV dot/accumulate with AVX2
- precomputed RoPE cos/sin tables
- no temporary per-token RoPE vectors during batched prefill
- hardware batch/thread auto-tuner (`memvanta_auto_tune`)

## Benchmark protocol

Environment:

- CPU allocation: 5 vCPUs
- Host CPU: AMD EPYC 9V74 80-Core Processor
- RAM: ~5.9 GiB
- Swap: none
- compiler: GCC/C++20, Release, `-O3 -march=native`
- model workload: deterministic SmolLM2-135M-shape GGUF systems fixture, 134.52M tensor elements, 85.77 MiB

Protocol mirrors llama.cpp `llama-bench` defaults where applicable:

- pp512
- tg128
- 5 measured repetitions
- warm-up enabled
- CPU-only
- fixed thread count per comparison
- mean ± sample standard deviation

The fixture has the target model's transformer dimensions and quantized tensor layout but synthetic weights. Therefore these results measure the MemVanta runtime and are **not** claimed as trained-model quality or a direct llama.cpp victory.

## Apples-to-apples v0.5 → v0.6 comparison

The exact v0.5 primary settings were reused:

- 5 threads
- batch = 64
- FP32 KV cache
- pp512 / tg128
- 5 repetitions + warm-up

| Version | pp512 | tg128 | Peak RSS |
|---|---:|---:|---:|
| v0.5 | 133.60 ± 18.85 tok/s | 67.57 ± 4.66 tok/s | ~124 MiB |
| **v0.6** | **322.42 ± 32.64 tok/s** | **67.79 ± 5.94 tok/s** | **124.65 MiB** |
| Change | **2.41× / +141.3%** | **1.003× / +0.3%** | essentially unchanged |

This is the key v0.6 result: prompt processing is about 2.4× faster at the same benchmark settings while decode throughput is preserved.

## Compact-KV run

With the tuner-selected direction of 4 threads, batch 32, F16 KV:

| Test | Result |
|---|---:|
| pp512 | 286.56 ± 26.43 tok/s |
| tg128 | 92.33 ± 12.28 tok/s |
| peak RSS | ~112.20 MiB |
| allocated KV pages | 5.62 MiB |

The lower-RSS F16 run is useful for memory-constrained systems, while the apples-to-apples FP32 run above is the fair v0.5 comparison.

## Auto-tuner

`memvanta_auto_tune` measures combinations of CPU threads and prompt batches on the current host instead of assuming that more threads or a larger batch is always better.

A pp128 tuning pass found:

- best measured combination: 4 threads, batch 32
- measured tuning throughput: 415.668 tok/s

Because this is a shared virtualized CPU, tuning results vary with scheduler contention. The tool therefore emits raw CSV so the chosen point is auditable.

## Activation-Q8 path

v0.6 implements batch-wide Q8 activation quantization. Each 32-element activation block is quantized once and reused across all output rows. On this particular EPYC VM the register-blocked FP32-activation microkernel was faster, so it is the default. The Q8-activation path remains selectable with:

```bash
MEMVANTA_V06_Q8_ACT=1 ./memvanta_real_bench ...
```

This avoids forcing a theoretically attractive optimization when the local CPU benchmark says otherwise.

## Remaining gap before a public llama.cpp claim

The next external comparison must use the same real trained GGUF in both engines, the same quantization, CPU affinity/thread count, context/KV type, and pp512/tg128 protocol. Until that exact A/B run is available, MemVanta should claim only the measured v0.5→v0.6 runtime improvement shown above.
