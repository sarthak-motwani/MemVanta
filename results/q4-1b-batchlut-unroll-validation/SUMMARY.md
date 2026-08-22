# Q4 batch-LUT + unroll TinyLlama 1.1B validation

Exact TinyLlama 1.1B Q4_0; 5 randomized reps/variant; production source restored.

| Variant | Prefill tok/s | Decode tok/s | Peak RSS MiB | Prefill CV | Decode CV |
|---|---:|---:|---:|---:|---:|
| baseline | 16.47 | 5.92 | 590.78 | 1.56% | 0.74% |
| candidate | 16.88 | 5.99 | 590.87 | 0.49% | 0.27% |

Combined speed gain: **1.44%**.
RSS delta: **0.01%**.
Promotion result: **FAIL**.
Gate: >=5% combined speed gain, <=2% RSS increase, and <=5% timing CV for both baseline and candidate.
