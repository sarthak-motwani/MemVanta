# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; candidates still need positive evidence before promotion.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 27941.90 ± 108.63 | 28036.38 ± 82.00 | -0.34% |
| Prefill ms | 35790.33 ± 56.05 | 35793.33 ± 144.71 | -0.01% |
| Decode ms | 9751.48 ± 6.41 | 9775.12 ± 25.82 | -0.24% |
| QKV ms | 10360.80 ± 70.07 | 10293.33 ± 38.39 | 0.65% |

- Total improvement: -0.06%
- FFN improvement: -0.34%
- RSS growth: -0.00%
- Gate: **PASS**
