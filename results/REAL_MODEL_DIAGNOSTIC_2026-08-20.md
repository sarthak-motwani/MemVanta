# First trained-model diagnostic A/B — 2026-08-20

This file preserves the first successful bounded real-model MemVanta vs llama.cpp CI result for later analysis. The values below were captured from the GitHub Actions `Real Model Validation` diagnostic summary.

## Configuration

- Model: `stories15M-q4_0.gguf` (trained TinyStories GGUF, Q4_0)
- CPU-only comparison
- 4 threads
- context: 128
- prompt: 16 tokens
- generation: 8 tokens
- repetitions: 1
- warm-up: 0
- MemVanta batch: 8
- MemVanta KV: F16
- client workload: disabled

## Results

| Metric | MemVanta | llama.cpp | Analysis |
|---|---:|---:|---:|
| Prompt processing | 1701.71 tok/s | 6561.682481 tok/s | MemVanta/llama.cpp = 0.25934x; llama.cpp ≈ 3.86x faster |
| Token generation | 616.348 tok/s | 1831.953948 tok/s | MemVanta/llama.cpp = 0.33644x; llama.cpp ≈ 2.97x faster |
| Peak RSS | 30,896 KiB | 43,164 KiB | MemVanta uses ≈ 28.4% less peak RSS |

Exit status:

- MemVanta benchmark: 0
- MemVanta greedy generation: 0
- llama.cpp benchmark: 0

## Interpretation

This run establishes that MemVanta can load and execute the same public trained GGUF used by llama.cpp and complete both benchmark and greedy-generation paths in CI. The most interesting preliminary observation is peak-memory use: MemVanta consumed about 12,268 KiB less peak RSS, roughly a 28.4% reduction relative to llama.cpp on this diagnostic run.

The throughput result currently favors llama.cpp substantially. That is expected for a mature, heavily optimized runtime and does not invalidate MemVanta's memory-adaptive systems thesis. The next question is whether MemVanta can preserve a meaningful memory advantage under repeated measurements and, later, under constrained-memory workloads where model size approaches or exceeds available RAM.

## Statistical boundary

These figures are diagnostic only. They use one repetition and no warm-up, so they should not be presented as stable comparative performance results. The promoted pp64/tg64 protocol uses one warm-up and five measured repetitions and should be used for publishable performance analysis.
