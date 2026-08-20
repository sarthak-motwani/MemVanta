# MemVanta v0.7.2 — real-model A/B validation

Target checkpoint: `ggml-org/models-moved` → `tinyllamas/stories15M-q4_0.gguf`, a trained TinyStories Llama GGUF Q4_0 checkpoint.

## Diagnostic gate — completed

A bounded CI diagnostic first established that both engines can execute the exact same trained GGUF successfully on CPU.

Configuration:

- context 128
- 4 logical CPU threads
- pp16 / tg8
- 1 measured repetition
- no warm-up
- MemVanta batch 8, F16 KV
- CPU only; llama.cpp uses `-ngl 0`
- client-style workload disabled

Observed CI summary:

| Metric | MemVanta | llama.cpp | Ratio |
|---|---:|---:|---:|
| Prompt processing | 1701.71 tok/s | 6561.682481 tok/s | 0.25934x |
| Token generation | 616.348 tok/s | 1831.953948 tok/s | 0.33644x |
| Peak RSS | 30,896 KiB | 43,164 KiB | 0.7158x |

Both benchmark commands exited with status 0. On this single diagnostic repetition, MemVanta used about 28.4% less peak RSS, while llama.cpp was about 3.86x faster in prompt processing and about 2.97x faster in token generation. These are diagnostic observations only, not stable performance claims, because the run used one repetition and no warm-up.

## Repeated performance gate

The CI workflow now promotes the trained-model comparison to the intended repeated benchmark:

- exact same GGUF file for both engines
- CPU only (`-ngl 0` for llama.cpp)
- identical logical CPU thread count: 4
- context 128
- pp64 / tg64
- 1 warm-up + 5 measured repetitions for MemVanta
- 5 repetitions in llama-bench
- MemVanta batch 32, F16 KV
- peak RSS captured with `/usr/bin/time -v`
- deterministic greedy generation captured from both engines
- both benchmark and greedy commands must exit successfully for a green A/B job
- raw benchmark outputs preserved as CI artifacts
- summary reports mean, sample SD, median, min, max, throughput ratios, and RSS reduction

## Why pp64/tg64, not pp512/tg128

The TinyStories 15M checkpoint has a native context length of 128. A pp512 test would exceed the model context. pp64/tg64 fills the native 128-token context and is therefore the full-context performance gate for this checkpoint. Larger-context models should use the conventional pp512/tg128 suite.

## Claim boundary

The completed pp16/tg8 diagnostic establishes successful trained-model execution and a preliminary memory observation. Publishable relative-performance conclusions should be based on the repeated pp64/tg64 artifact, not the one-repetition diagnostic result.
