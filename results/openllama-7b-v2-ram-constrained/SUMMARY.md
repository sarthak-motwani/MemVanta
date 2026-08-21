# OpenLLaMA 7B v2 RAM-constrained experiment

Linux cgroup-v2 `MemoryMax` sweep with swap disabled. Exact same Q4_0 GGUF, CPU-only, 4 threads, pp128/tg32, context 768, batch 32, F16 KV.

- MemVanta lowest successful tested limit: 3584 MiB
- llama.cpp lowest successful tested limit: 3840 MiB
- Tested-ceiling difference: 256 MiB
- MemVanta lower successful tested ceiling: 6.67%

This is an execution-under-pressure test over the stated sweep, not an exact minimum physical-RAM requirement and not a throughput benchmark.
