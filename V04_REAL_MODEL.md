# MemVanta CPU v0.4 — Full Real-Model Inference Gate

Date: 2026-08-20

## What v0.4 implements

MemVanta now contains two complete CPU decoder paths:

1. **Native llama2.c checkpoint path** (`Llama2Model`) for Karpathy-style FP32 checkpoints.
   - memory-mapped weights
   - token embeddings
   - RMSNorm
   - Q/K/V projections
   - rotary position embeddings (RoPE)
   - grouped-query / multi-query attention
   - causal softmax attention
   - KV cache
   - attention output projection and residual
   - SwiGLU feed-forward network and residual
   - final RMSNorm and LM head
   - greedy autoregressive decoding

2. **Native GGUF path** (`LlamaModel`) with:
   - GGUF v2/v3 metadata/tensor parsing
   - mmap-backed tensor access
   - GGUF tokenizer support
   - paged KV cache
   - F32/F16/Q4_0/Q8_0 tensor kernels (plus byte-size parsing for common K-quants)
   - text prompt -> tokens -> transformer -> sampler -> decoded text

The code compiles with GCC 14 / C++20 and the supplied CTest suite passes.

## Benchmark protocol

The benchmark names and repetition policy intentionally mirror llama.cpp `llama-bench`:

- prompt processing: `pp512`
- token generation: `tg128`
- 5 measured repetitions after warm-up
- report mean ± sample standard deviation
- fixed thread count
- report peak resident memory
- also report TTFT and output TPS as client-facing latency/throughput metrics

### Test host

- CPU: AMD EPYC 9V74 (virtualized)
- available vCPU: 5
- Linux x86-64
- compiler: GCC 14.2
- C++20, Release, `-O3 -march=native`
- RAM: about 5.9 GiB; no swap in earlier capacity tests

## Full-graph structural benchmark

A deterministic **1,056,540-byte checkpoint fixture** was generated with exactly the TinyStories-260K architecture and llama2.c binary layout:

- dim: 64
- hidden dim: 172
- layers: 5
- query heads: 8
- KV heads: 4
- vocabulary: 512
- context: 512

Important: the fixture contains deterministic generated weights, **not Karpathy's trained TinyStories weights**. It validates and times the full transformer execution path, but it is not published as a trained-model benchmark.

### 1 thread, 5 measured repetitions

| Metric | Result |
|---|---:|
| pp512 | **4261.80 ± 46.09 tok/s** |
| tg128 | **6551.55 ± 136.74 tok/s** |
| TTFT (128-token input) | **19.85 ± 0.87 ms** |
| output TPS | **4810.35 ± 151.77 tok/s** |
| peak RSS (`getrusage`) | ~5 MiB |

### 5 threads, 5 measured repetitions

| Metric | Result |
|---|---:|
| pp512 | **202.64 ± 15.79 tok/s** |
| tg128 | **226.64 ± 25.72 tok/s** |
| TTFT (128-token input) | **598.19 ± 99.46 ms** |
| output TPS | **219.45 ± 32.72 tok/s** |
| peak RSS (`getrusage`) | ~5 MiB |

The tiny architecture is too small for the current per-matmul `std::thread` creation strategy: 5 threads are much slower than one. This is a real and useful finding, not hidden benchmark noise. A persistent pool / OpenMP threshold policy is the next CPU scheduling optimization.

## Strict real-model benchmark gate

The standard target is Karpathy TinyStories-260K:

- `stories260K.bin`: 1,056,540 bytes
- SHA-256: `b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696`
- tokenizer: `tok512.bin`
- tokenizer SHA-256: `e6e45b754b603ab1fb1a31e59c1ebbee92a789504c3ddf6debb6bd3c106222d6`

`llama.cpp`'s own CPU CI downloads this checkpoint/tokenizer, converts the llama2.c checkpoint to GGUF, and executes it. This makes it an excellent parity target.

The current sandbox could access public metadata and source documentation but could not materialize Hugging Face Xet/LFS binary objects into the local runtime. Therefore **no trained-weight MemVanta-vs-llama.cpp speedup claim is made in this report**. Publishing the fixture result as a real TinyStories result would be misleading.

## Exact apples-to-apples benchmark once the checkpoint is present

MemVanta native checkpoint path:

```bash
./build/memvanta_llama2c_bench \
  --model stories260K.bin \
  --threads 1 \
  --pp 512 --tg 128 --reps 5 --warmup 1
```

For llama.cpp, convert the exact same checkpoint using its supported llama2.c converter and `tok512.bin`, then run:

```bash
./llama-convert-llama2c-to-ggml \
  --copy-vocab-from-model tok512.bin \
  --llama2c-model stories260K.bin \
  --llama2c-output-model stories260K.gguf

./llama-bench -m stories260K.gguf -t 1 -p 512 -n 128 -r 5 -o csv
```

Repeat with the same CPU affinity at 2, 3, 4, and 5 threads. The comparison table should not be filled until both programs run on the same host and exact trained checkpoint.

## Industry comparison fields to publish

| Field | MemVanta | llama.cpp |
|---|---:|---:|
| checkpoint SHA-256 | same | same source checkpoint |
| quantization | same | same |
| CPU / ISA | same | same |
| threads / affinity | same | same |
| context | same | same |
| pp512 mean ± SD | pending trained weights | pending |
| tg128 mean ± SD | pending trained weights | pending |
| TTFT | pending | pending |
| output TPS | pending | pending |
| peak RSS | pending | pending |
| load/init time | pending | pending |
| deterministic output parity | pending | reference |

## Interpretation

v0.4 crosses the engineering boundary from a streaming/kernel prototype to a complete CPU transformer execution runtime. The remaining blocker to an externally defensible `MemVanta vs llama.cpp` number in this sandbox is **artifact availability**, not absence of the inference graph. Until the exact trained checkpoint can be materialized locally, the repository should describe the benchmark above as **full-graph structural validation** and not as a trained-model performance result.
