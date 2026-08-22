# Q4 LUT SmolLM2-360M A/B

Exact SmolLM2-360M Q4_0; 5 alternating reps/variant; production source restored.

| Variant | Prefill tok/s | Decode tok/s | Peak RSS MiB | Prefill CV | Decode CV |
|---|---:|---:|---:|---:|---:|
| baseline | 48.35 | 30.41 | 248.82 | 0.52% | 0.64% |
| lut | 48.25 | 30.59 | 248.82 | 0.42% | 0.10% |

Combined speed gain: **0.28%**.
RSS delta: **0.00%**.
Promotion result: **FAIL**.
Gate: >=5% combined speed gain, <=2% RSS increase, and <=5% timing CV for both baseline and candidate.
