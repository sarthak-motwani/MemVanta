# MemVanta CPU v0.7 — real-model A/B protocol

v0.7 adds the reproducible headline benchmark harness for **the exact same trained GGUF** in MemVanta and current llama.cpp.

Metrics: llama-bench-compatible `pp512` and `tg128`; MemVanta TTFT and output TPS; `/usr/bin/time -v` peak RSS; model-load latency; SHA-256 and exact byte size; deterministic greedy output; and optional cross-entropy/perplexity on a shared text corpus.

## Fairness rules

- Same GGUF file and SHA-256 for both engines.
- CPU only (`-ngl 0` for llama.cpp).
- Same logical CPU allocation and thread count.
- Same 512-token prompt / 128-token generation headline workload, 1 warm-up, 5 measured repetitions.
- Report mean and standard deviation; keep outliers unless a predeclared system-level exclusion applies.
- Report KV format and context explicitly.
- Do **not** call synthetic/model-shape fixtures "real-model" results.

Recommended trained checkpoint: `QuantFactory/SmolLM2-135M-Instruct-GGUF`, file `SmolLM2-135M-Instruct.Q4_0.gguf`, 91,726,912 bytes, SHA-256 `f68203bfb98b1b1e8c64fac75fab10c4e36acac081609573b4df0fcc19c90dd9`.

Run:

```bash
cmake -S . -B build-v07 -DCMAKE_BUILD_TYPE=Release
cmake --build build-v07 -j
./scripts/run_v07_headline.sh /path/model.gguf 5 /path/to/llama.cpp/build/bin /path/to/corpus.txt
```
