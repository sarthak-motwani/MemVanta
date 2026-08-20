# MemVanta v0.7.2 — first real-model A/B gate

Target checkpoint: `ggml-org/models` → `tinyllamas/stories15M-q4_0.gguf`, a trained TinyStories Llama GGUF Q4_0 checkpoint.

## Why pp64/tg64, not pp512/tg128

The current official 15M checkpoint has a native context length of 128. A 512-token prompt benchmark would be invalid for this checkpoint. v0.7.2 therefore uses `pp64` and `tg64` with context 128 for the first genuine correctness/performance A/B. Larger-context models should retain the usual pp512/tg128 suite.

## Fairness controls

- exact same GGUF file for both engines
- CPU only (`-ngl 0` for llama.cpp)
- identical logical CPU thread count
- 1 warm-up + 5 measured repetitions for MemVanta; 5 reps in llama-bench
- F16 KV mode in MemVanta
- peak RSS captured with `/usr/bin/time -v`
- deterministic greedy generation captured from both engines
- raw benchmark outputs preserved as CI artifacts

## Execution status in the ChatGPT CPU sandbox

The sandbox cannot resolve external DNS and cannot download Hugging Face model bytes. The source builds and CTest passes locally, but the trained checkpoint cannot be materialized here. The included GitHub Actions workflow is intentionally the execution route because GitHub-hosted runners have network access. No MemVanta-vs-llama.cpp performance claim is made until that workflow has produced raw outputs.
