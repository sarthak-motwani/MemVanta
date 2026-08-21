# SmolLM2-360M Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, KV=f16, 5 repetitions.

- MemVanta pp: 46.62 ± 0.92 tok/s
- llama.cpp pp: 186.98 ± 0.77 tok/s
- MemVanta tg: 24.83 ± 0.07 tok/s
- llama.cpp tg: 110.88 ± 0.22 tok/s
- MemVanta peak RSS: 266700 KiB
- llama.cpp peak RSS: 448936 KiB
- MemVanta RSS reduction: 40.59%
