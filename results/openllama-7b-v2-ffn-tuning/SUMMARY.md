# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | QKV ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 47367.70 | 11423.60 | 36414.47 | 13702.76 | 58791.30 |
| fp32 | 32 | 48676.40 | 11431.50 | 38015.63 | 13586.55 | 60107.90 |
| fp32 | 64 | 55024.60 | 11505.80 | 41297.95 | 15944.68 | 66530.40 |
| q8act | 64 | 136268.00 | 11397.00 | 95901.23 | 35797.31 | 147665.00 |
| q8act | 32 | 136255.00 | 11479.70 | 95923.43 | 35808.75 | 147734.70 |
| q8act | 16 | 136435.00 | 11429.30 | 95909.23 | 35863.10 | 147864.30 |

Best measured configuration: **fp32 activation path, batch 16**.

This is a tuning experiment, not a llama.cpp comparison claim.
