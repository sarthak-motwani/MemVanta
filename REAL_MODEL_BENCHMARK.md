# MemVanta CPU v0.4 — real GGUF inference benchmark status

## What is implemented

The v0.4 path is a standalone C++20 Llama-family GGUF decoder. It includes:

- mmap-backed GGUF v2/v3 metadata and tensor parser
- zero-copy F32/F16/Q4_0/Q8_0 tensor access
- AVX2/FMA Q4_0 and Q8_0 matrix-vector kernels with OpenMP row parallelism
- learned RMSNorm
- Llama non-interleaved RoPE
- grouped-query attention (GQA)
- lazy paged FP32 KV cache
- SwiGLU feed-forward network
- tied or explicit LM output head
- GGUF GPT-2/BPE tokenizer and deterministic sampler
- end-to-end autoregressive generation
- llama-bench-style `pp512` / `tg128` benchmark harness

`memvanta_real` is the inference CLI and `memvanta_real_bench` is the benchmark executable.

## Industry-style benchmark protocol

For the comparable CPU throughput test we use llama.cpp's conventional defaults:

- prompt processing: **512 tokens** (`pp512`)
- generation: **128 tokens** (`tg128`)
- **5 repetitions**
- warm-up before measured runs
- fixed CPU thread count
- same GGUF file and quantization for both engines
- CPU-only execution
- report arithmetic mean and sample standard deviation
- keep tokenization and sampling outside timed `pp`/`tg` sections

For a publishable MemVanta-vs-llama.cpp number, the public checkpoint's SHA256 must match on both sides.

## Target public checkpoint

The current compatibility target is:

- repository: `QuantFactory/SmolLM2-135M-Instruct-GGUF`
- file: `SmolLM2-135M-Instruct.Q4_0.gguf`
- architecture: Llama
- parameters: about 134.5M
- public file size: 91.7 MB
- SHA256: `f68203bfb98b1b1e8c64fac75fab10c4e36acac081609573b4df0fcc19c90dd9`

This quant was selected because its dense block tensors use the Q4_0/Q8_0/F32 paths implemented in v0.4.

## Full-shape execution validation on this host

The external checkpoint binary could not be downloaded in this sandbox because the Hugging Face Xet binary redirect was unavailable to the runtime. A remote Hugging Face CPU job was also unavailable (payment-required response). Therefore **no public-checkpoint or llama.cpp-vs-MemVanta performance claim is fabricated here**.

To validate the complete inference path anyway, `scripts/make_smollm2_shape_gguf.py` creates a valid GGUF v3 systems fixture with the exact SmolLM2-135M dimensions and key quantized tensor types:

- embedding width 576
- FFN width 1536
- 30 transformer blocks
- 9 attention heads / 3 KV heads
- vocabulary 49,152
- Q4_0 transformer matrices
- Q8_0 token embedding
- F32 norm vectors
- 134.52M tensor elements

The values are deterministic synthetic values, **not trained SmolLM2 weights**. Consequently this validates execution, memory behavior and shape-equivalent systems throughput only; it is not a model-quality benchmark.

### Environment

- CPU: AMD EPYC 9V74
- allocated CPU cores: 5
- RAM: ~5.9 GiB
- swap: 0
- ISA exposed: AVX2/FMA and AVX-512 family
- backend measured: MemVanta C++ CPU
- threads: 5

### Five independent measured runs

The same `pp512` and `tg128` workload was executed five times. A previous execution had already faulted the fixture into the Linux page cache, so these are warm-file-cache runs. Each process used the same 5-thread configuration.

| run | pp512 tok/s | tg128 tok/s |
|---:|---:|---:|
| 1 | 52.02 | 60.86 |
| 2 | 56.30 | 60.99 |
| 3 | 55.29 | 45.52 |
| 4 | 27.34 | 52.33 |
| 5 | 57.28 | 43.74 |
| **mean ± sample SD** | **49.64 ± 12.62** | **52.69 ± 8.18** |
| **median** | **55.29** | **52.33** |

Peak RSS across these runs was approximately **122.25 MiB**. The Q4/Q8 tensor payload exposed by the parser is about **85.77 MiB**; the complete fixture is about **86.62 MiB** including GGUF metadata/token strings.

Run 4 shows substantial host scheduling interference. It is deliberately retained rather than discarded. This sandbox is shared/virtualized, so these figures are engineering measurements, not controlled bare-metal submissions.

## What this result does and does not establish

It establishes that the new v0.4 engine can parse and execute an end-to-end Llama-family transformer with the **same dimensionality, layer count, GQA topology, vocabulary scale and relevant Q4_0/Q8_0 tensor formats** as the chosen public SmolLM2 checkpoint. It also establishes that the benchmark harness can emit the same headline test labels used by `llama-bench`.

It does **not** yet establish that MemVanta is faster than llama.cpp. That claim requires the exact trained GGUF checkpoint plus `llama-bench` on this same host. The included `scripts/run_real_compare.sh` performs that apples-to-apples run once both are locally available.

## Reproduce

```bash
cmake -S . -B build-real -DCMAKE_BUILD_TYPE=Release
cmake --build build-real -j
ctest --test-dir build-real --output-on-failure

# Shape-equivalent systems fixture
python3 scripts/make_smollm2_shape_gguf.py /tmp/smollm2-shape.gguf
./build-real/memvanta_real_bench \
  --model /tmp/smollm2-shape.gguf --threads 5 --ctx 1024 \
  --prompt 512 --gen 128 --warmup 1 --reps 5 --no-client \
  --csv results.csv

# Exact public-model A/B, after the real model and llama-bench are present:
./scripts/run_real_compare.sh /path/to/SmolLM2-135M-Instruct.Q4_0.gguf 5 /path/to/llama-bench
```

## Next performance work

The largest performance gap is prompt processing. MemVanta v0.4 currently evaluates prefill token-by-token. llama.cpp batches prompt tokens and uses optimized matrix-matrix kernels, so a fair speed contest should not be expected until MemVanta adds batched prefill. The immediate v0.5 targets are batched GEMM prefill, vectorized attention, a worker-pool that avoids per-matvec OpenMP launch overhead, F16/Q8 KV-cache options, and exact tokenizer parity tests against llama.cpp.
