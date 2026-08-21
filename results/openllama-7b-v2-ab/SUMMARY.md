# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 3.15 ± 0.04 tok/s
- llama.cpp pp: 45.82 ± 0.96 tok/s
- MemVanta tg: 1.52 ± 0.02 tok/s
- llama.cpp tg: 7.76 ± 0.09 tok/s
- MemVanta peak RSS: 3983412 KiB
- llama.cpp peak RSS: 7586960 KiB
- MemVanta RSS reduction: 47.50%
