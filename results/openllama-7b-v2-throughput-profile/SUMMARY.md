# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- ffn_gemm_ms: 28321.49 ms (58.9% of profiled component time)
- qkv_projection_ms: 11972.83 ms (24.9% of profiled component time)
- attention_output_projection_ms: 3988.39 ms (8.3% of profiled component time)
- output_head_ms: 2748.29 ms (5.7% of profiled component time)
- non_gemm_core_ms: 1075.99 ms (2.2% of profiled component time)

This profile is for hotspot selection, not a throughput comparison claim.
