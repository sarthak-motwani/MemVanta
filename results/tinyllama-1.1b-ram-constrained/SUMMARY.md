# TinyLlama 1.1B RAM-constrained experiment

Linux cgroup-v2 MemoryMax sweep with swap disabled. Same Q4_0 GGUF, CPU-only, 4 threads, pp128/tg32, context 768, batch 32, F16 KV.

- MemVanta lowest successful limit: none MiB
- llama.cpp lowest successful limit: none MiB

This is an execution-under-pressure boundary test, not a throughput benchmark. Full per-limit outcomes are in `results.csv`.
