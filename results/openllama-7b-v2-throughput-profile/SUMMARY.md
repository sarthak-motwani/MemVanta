# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 47415.30 ms
- Decode total: 12803.50 ms
- Projection kernels: 59142.91 ms (98.2% of profiled model time)
- FFN GEMM: 36928.12 ms (62.4% of projection-kernel time)
- Non-GEMM core residual: 1075.32 ms
- Dominant component: ffn_gemm_ms (36928.12 ms)
- Dominant profiled kernel kind: ffn_down (12649.05 ms)
- Peak RSS: 3810612 KiB
- Page faults: major=1, minor=96104
- Process CPU: 239%

This profile selects the next optimization target; it is not a universal throughput claim.
