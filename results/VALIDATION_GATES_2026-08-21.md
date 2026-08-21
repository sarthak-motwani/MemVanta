# Validation gates — 2026-08-21

Both production validation workflows are green on `main` as of 2026-08-21.

## TinyStories 15M repeated A/B

Workflow: `Real Model Validation`

Validated configuration:

- trained TinyStories 15M Q4_0 GGUF
- CPU only
- 4 logical CPU threads
- context 128
- pp64 / tg64
- MemVanta batch 32
- F16 KV
- 1 warm-up + 5 measured repetitions
- pinned llama.cpp reference
- deterministic greedy-generation gate
- trained-model profiling and raw artifact upload

Status: **green**.

This establishes successful repeated same-model execution of MemVanta and the pinned llama.cpp reference under the declared benchmark protocol. Numerical throughput and RSS claims should continue to come from the preserved workflow artifact/summary rather than be inferred from the green status alone.

## SmolLM2-360M validation

Workflow: `Medium Model Validation`

Validated configuration:

- SmolLM2-360M Q4_0 GGUF
- SHA-256 `88bbd5cac17036bf50905a0a660078b63426cea1c7fc6e4afa955bc8422b043c`
- CPU only
- 4 logical CPU threads
- context 128
- tokenizer smoke test
- deterministic greedy-generation smoke test
- short real-model benchmark
- peak-memory/time evidence upload

Status: **green**.

This establishes successful native MemVanta execution on a trained 360M-parameter-class GGUF checkpoint, extending validation beyond the TinyStories 15M diagnostic model.

## Claim boundary

A green validation run establishes execution correctness and workflow robustness. It does not, by itself, establish a throughput or memory advantage over llama.cpp. Comparative performance claims must cite the repeated A/B numerical artifact, including mean, sample SD, median/min/max, coefficient of variation, throughput ratios, exit codes, and peak RSS.
