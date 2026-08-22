# Q4 360M top-2 validation

Exact SmolLM2-360M Q4_0; 5 randomized reps/variant; production source restored.

| Variant | Prefill tok/s | Decode tok/s | Combined gain | RSS delta | Prefill CV | Decode CV | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| baseline | 50.66 | 29.00 | 0.00% | 0.00% | 0.65% | 0.40% | no |
| unroll | 50.40 | 29.53 | 0.95% | 0.05% | 0.95% | 4.21% | no |
| batchlut_unroll | 56.07 | 30.25 | 6.54% | -0.02% | 0.20% | 0.16% | YES |

Winner: **batchlut_unroll**.
Promotion requires >=5% combined speed gain, <=2% RSS increase, and <=5% timing CV.
