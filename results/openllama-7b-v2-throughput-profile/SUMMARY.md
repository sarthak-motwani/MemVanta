# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- ffn_gemm_ms: 38932.39 ms (63.0% of profiled component time)
- qkv_projection_ms: 14052.00 ms (22.7% of profiled component time)
- attention_output_projection_ms: 4709.61 ms (7.6% of profiled component time)
- output_head_ms: 3043.50 ms (4.9% of profiled component time)
- non_gemm_core_ms: 1034.39 ms (1.7% of profiled component time)

This profile is for hotspot selection, not a throughput comparison claim.
