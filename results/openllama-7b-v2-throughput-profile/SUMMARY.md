# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- ffn_gemm_ms: 35601.44 ms (62.1% of profiled component time)
- qkv_projection_ms: 13205.83 ms (23.0% of profiled component time)
- attention_output_projection_ms: 4436.66 ms (7.7% of profiled component time)
- output_head_ms: 3044.52 ms (5.3% of profiled component time)
- non_gemm_core_ms: 1025.59 ms (1.8% of profiled component time)

This profile is for hotspot selection, not a throughput comparison claim.
