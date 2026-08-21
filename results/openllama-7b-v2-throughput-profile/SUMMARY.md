# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 16, F16 KV.

- ffn_gemm_ms: 36277.38 ms (61.8% of profiled component time)
- qkv_projection_ms: 13751.70 ms (23.4% of profiled component time)
- attention_output_projection_ms: 4572.11 ms (7.8% of profiled component time)
- output_head_ms: 3051.22 ms (5.2% of profiled component time)
- non_gemm_core_ms: 1045.37 ms (1.8% of profiled component time)

This profile is for hotspot selection, not a throughput comparison claim.
