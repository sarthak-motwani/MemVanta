# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; candidates still need positive evidence before promotion.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 36733.36 ± 212.50 | 36628.11 ± 143.51 | 0.29% |
| Prefill ms | 46767.33 ± 351.55 | 46647.90 ± 271.78 | 0.26% |
| Decode ms | 12800.40 ± 56.75 | 12757.57 ± 26.65 | 0.33% |
| QKV ms | 13769.63 ± 132.99 | 13756.73 ± 121.79 | 0.09% |

- Total improvement: 0.27%
- FFN improvement: 0.29%
- RSS growth: 0.00%
- Gate: **PASS**
