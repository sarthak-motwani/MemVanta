# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 17.12 ± 0.04 tok/s
- llama.cpp pp: 79.00 ± 0.28 tok/s
- MemVanta tg: 5.79 ± 0.00 tok/s
- llama.cpp tg: 48.58 ± 0.06 tok/s
- MemVanta peak RSS: 644100 KiB
- llama.cpp peak RSS: 1190124 KiB
- MemVanta RSS reduction: 45.88%
