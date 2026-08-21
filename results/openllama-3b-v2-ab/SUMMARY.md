# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 5.28 ± 0.03 tok/s
- llama.cpp pp: 24.46 ± 0.02 tok/s
- MemVanta tg: 3.43 ± 0.00 tok/s
- llama.cpp tg: 15.67 ± 0.04 tok/s
- MemVanta peak RSS: 2120208 KiB
- llama.cpp peak RSS: 3885236 KiB
- MemVanta RSS reduction: 45.43%
