# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- ffn_gemm_ms: 35677.82 ms (62.0% of profiled component time)
- qkv_projection_ms: 13305.92 ms (23.1% of profiled component time)
- attention_output_projection_ms: 4432.80 ms (7.7% of profiled component time)
- output_head_ms: 3047.76 ms (5.3% of profiled component time)
- non_gemm_core_ms: 1036.20 ms (1.8% of profiled component time)

This profile is for hotspot selection, not a throughput comparison claim.
