# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 6.62 ± 0.06 tok/s
- llama.cpp pp: 33.59 ± 0.05 tok/s
- MemVanta tg: 4.09 ± 0.01 tok/s
- llama.cpp tg: 17.40 ± 0.09 tok/s
- MemVanta peak RSS: 2120428 KiB
- llama.cpp peak RSS: 3885332 KiB
- MemVanta RSS reduction: 45.42%
