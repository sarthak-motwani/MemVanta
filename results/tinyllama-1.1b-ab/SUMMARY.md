# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 17.12 ± 0.08 tok/s
- llama.cpp pp: 79.13 ± 0.13 tok/s
- MemVanta tg: 5.77 ± 0.01 tok/s
- llama.cpp tg: 48.03 ± 0.12 tok/s
- MemVanta peak RSS: 644240 KiB
- llama.cpp peak RSS: 1190208 KiB
- MemVanta RSS reduction: 45.87%
