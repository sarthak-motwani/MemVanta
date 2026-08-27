# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 32 | 35595.70 | 9801.47 | 27873.26 | 45397.17 | 3810688 |
| fp32 | 64 | 37800.60 | 9755.24 | 29325.66 | 47555.84 | 3810680 |
| fp32 | 16 | 38474.00 | 9777.67 | 29391.62 | 48251.67 | 3811040 |
| q8act | 64 | 115466.00 | 9746.32 | 81094.78 | 125212.32 | 3810988 |
| q8act | 32 | 115540.00 | 9774.16 | 81117.36 | 125314.16 | 3810664 |
| q8act | 16 | 115804.00 | 9755.05 | 81238.55 | 125559.05 | 3811052 |

Best FP32: batch 32; best Q8-act: batch 64.
Q8-act end-to-end improvement vs best FP32: **-175.82%**.
Q8-act FFN improvement vs best FP32: **-190.94%**.
Q8-act peak-RSS delta: **0.01%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
