# Q4 nibble unpack screen

Exact TinyStories 15M Q4_0; 7 randomized reps/variant; production source restored.

| Variant | Prefill tok/s | Decode tok/s | Peak RSS MiB | Combined gain | RSS delta | Prefill CV | Decode CV | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| baseline | 1827.13 | 571.58 | 26.15 | 0.00% | 0.00% | 2.79% | 27.36% | no |
| lut | 1874.38 | 642.20 | 26.23 | 9.86% | 0.34% | 2.56% | 0.79% | YES |
| bias32 | 1852.76 | 640.02 | 26.18 | 9.26% | 0.15% | 3.21% | 4.56% | YES |

Winner: **lut**.
Promotion gate: >=5% combined speed gain, <=2% RSS increase, <=5% timing CV.
