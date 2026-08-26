# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 16.80 ± 0.02 tok/s
- llama.cpp pp: 78.64 ± 0.46 tok/s
- MemVanta tg: 5.75 ± 0.02 tok/s
- llama.cpp tg: 47.78 ± 0.57 tok/s
- MemVanta peak RSS: 644332 KiB
- llama.cpp peak RSS: 1190252 KiB
- MemVanta RSS reduction: 45.87%
