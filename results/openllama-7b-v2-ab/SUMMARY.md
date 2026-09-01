# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 2.72 ± 0.01 tok/s
- llama.cpp pp: 11.96 ± 0.00 tok/s
- MemVanta tg: 1.25 ± 0.00 tok/s
- llama.cpp tg: 7.83 ± 0.11 tok/s
- MemVanta peak RSS: 3983188 KiB
- llama.cpp peak RSS: 7590720 KiB
- MemVanta RSS reduction: 47.53%
