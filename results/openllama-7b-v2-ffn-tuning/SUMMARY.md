# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 47348.10 | 11458.00 | 36390.27 | 58806.10 | 3810864 |
| fp32 | 32 | 49696.00 | 11431.10 | 38189.58 | 61127.10 | 3810460 |
| fp32 | 64 | 49955.30 | 11445.60 | 38028.02 | 61400.90 | 3810640 |
| q8act | 32 | 136216.00 | 11421.10 | 95807.06 | 147637.10 | 3810664 |
| q8act | 64 | 136331.00 | 11433.70 | 95949.65 | 147764.70 | 3810740 |
| q8act | 16 | 136367.00 | 11399.40 | 95822.69 | 147766.40 | 3811012 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-151.06%**.
Q8-act FFN improvement vs best FP32: **-163.28%**.
Q8-act peak-RSS delta: **-0.01%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
