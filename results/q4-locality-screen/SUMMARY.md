# Q4 locality kernel screen

Exact TinyStories 15M Q4_0; 5 randomized reps/variant; production source unchanged.

| Variant | Prefill tok/s | Decode tok/s | Peak RSS MiB | Combined gain | RSS delta | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| baseline | 2431.74 | 788.16 | 26.31 | 0.00% | 0.00% | no |
| pf1 | 2416.92 | 536.27 | 26.31 | -26.27% | 0.01% | no |
| pf2 | 2373.93 | 797.88 | 26.35 | 0.33% | 0.15% | no |
| pf4 | 2307.86 | 769.68 | 26.42 | -3.03% | 0.42% | no |
| pf8 | 2341.59 | 687.94 | 26.32 | -10.67% | 0.03% | no |

Winner: **none**.
Promotion gate: >=5% combined speed gain, <=2% RSS increase, <=5% timing CV.
