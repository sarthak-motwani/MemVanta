# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 47916.80 ms
- Decode total: 11413.80 ms
- Projection kernels: 58300.53 ms (98.3% of profiled model time)
- FFN GEMM: 37487.04 ms (64.3% of projection-kernel time)
- Non-GEMM core residual: 1029.64 ms
- Dominant component: ffn_gemm_ms (37487.04 ms)
- Dominant profiled kernel kind: ffn_down (13563.39 ms)
- Peak RSS: 3810668 KiB
- Page faults: major=1, minor=96104
- Process CPU: 236%

This profile selects the next optimization target; it is not a universal throughput claim.
