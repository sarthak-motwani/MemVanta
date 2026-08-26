# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 4.74 ± 0.02 tok/s
- llama.cpp pp: 28.05 ± 0.16 tok/s
- MemVanta tg: 3.39 ± 0.02 tok/s
- llama.cpp tg: 11.16 ± 0.06 tok/s
- MemVanta peak RSS: 2119516 KiB
- llama.cpp peak RSS: 3885032 KiB
- MemVanta RSS reduction: 45.44%
