# OpenLLaMA 7B FFN kernel same-runner A/B

Exact same runner, exact same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

| Metric | Baseline | Candidate | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 37054.35 | 36654.45 | 1.08% |
| Prefill ms | 48483.80 | 48102.20 | 0.79% |
| Decode ms | 11372.00 | 11387.60 | -0.14% |
| QKV ms | 14065.55 | 14103.11 | -0.27% |

Positive improvement means lower elapsed time for the candidate kernel.
