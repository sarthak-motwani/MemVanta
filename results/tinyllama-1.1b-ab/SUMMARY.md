# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.39 ± 0.09 tok/s
- llama.cpp pp: 107.90 ± 0.48 tok/s
- MemVanta tg: 5.58 ± 0.07 tok/s
- llama.cpp tg: 52.13 ± 0.60 tok/s
- MemVanta peak RSS: 644260 KiB
- llama.cpp peak RSS: 1194372 KiB
- MemVanta RSS reduction: 46.06%
