# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | QKV ms | Total ms |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 43436.40 | 11441.80 | 33779.45 | 12716.59 | 54878.20 |
| fp32 | 64 | 47694.00 | 11455.60 | 36597.90 | 13867.81 | 59149.60 |
| fp32 | 32 | 48636.40 | 11432.20 | 37184.28 | 14138.86 | 60068.60 |
| q8act | 64 | 136241.00 | 11391.40 | 95873.62 | 35800.27 | 147632.40 |
| q8act | 32 | 136233.00 | 11405.90 | 95857.40 | 35793.37 | 147638.90 |
| q8act | 16 | 136491.00 | 11425.20 | 95942.42 | 35887.12 | 147916.20 |

Best measured configuration: **fp32 activation path, batch 16**.

This is a tuning experiment, not a llama.cpp comparison claim.
